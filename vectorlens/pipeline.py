"""Auto-attribution pipeline.

Subscribes to llm_response events on the bus and automatically runs
hallucination detection + chunk attribution, storing results back to the bus.
Uses a bounded ThreadPoolExecutor — prevents thread explosion under load.
Conditionally triggers deep attribution only when hallucinations are detected.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, Tuple

from vectorlens.types import (
    AttributionResult,
    CAGContextEvent,
    GraphRAGContextEvent,
    LLMResponseEvent,
    RetrievedChunk,
    TokenHeatmapEntry,
)

logger = logging.getLogger(__name__)

_attribution_lock = threading.Lock()
_installed = False

# Bounded pool: max 3 concurrent attribution workers.
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vectorlens-attr")

# Bounded pending queue: max 50 tasks waiting for a free worker.
# ThreadPoolExecutor uses an unbounded SimpleQueue internally — without a
# cap, high-throughput LLM calls pile up in memory until OOM.
# Tasks are DROPPED (not blocked) when full; attribution is best-effort.
_MAX_PENDING = 50
_pending_sem = threading.Semaphore(_MAX_PENDING)


def setup_auto_attribution() -> None:
    """Subscribe to the bus and run attribution after every LLM response."""
    global _installed
    with _attribution_lock:
        if _installed:
            return
        from vectorlens.session_bus import bus
        bus.subscribe("llm_response", _on_llm_response)
        _installed = True
        logger.info("Auto-attribution pipeline active (max_workers=3, max_pending=50)")


def _on_llm_response(event: LLMResponseEvent) -> None:
    """Triggered after every LLM response — submits to bounded pool.

    Drops the task if the pending queue is full rather than letting
    the queue grow without bound and exhaust memory.
    """
    if not _pending_sem.acquire(blocking=False):
        logger.debug("Attribution queue full (>50 pending), dropping task")
        return

    def _run_and_release() -> None:
        try:
            _run_attribution(event)
        finally:
            _pending_sem.release()

    _executor.submit(_run_and_release)


def _try_get_hf_model_and_tokenizer(
    session: Any, response_event: LLMResponseEvent
) -> Tuple[Optional[Any], Optional[Any]]:
    """Retrieve the HF model + tokenizer for this session.

    Reads from the session object (set by TransformersInterceptor) rather than
    a global variable, so concurrent sessions each get their own model.

    Returns (model, tokenizer) if available, else (None, None).
    """
    try:
        model = getattr(session, "hf_model", None)
        tokenizer = getattr(session, "hf_tokenizer", None)
        if model is not None and tokenizer is not None:
            return model, tokenizer
    except Exception:
        pass
    return None, None


def _make_llm_caller(provider: str, model: str) -> Optional[Callable]:
    """Build an async LLM caller for LIME perturbation based on provider.

    Returns None if the provider is unsupported or SDK not installed.
    The caller signature: async (messages: list[dict]) -> str
    """
    if provider == "openai":
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                return None
            client = openai.AsyncOpenAI(api_key=api_key)

            async def _call_openai(messages: list[dict]) -> str:
                resp = await client.chat.completions.create(
                    model=model or "gpt-4o-mini",
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0,
                )
                return resp.choices[0].message.content or ""

            return _call_openai
        except ImportError:
            return None

    if provider == "anthropic":
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return None
            client = anthropic.AsyncAnthropic(api_key=api_key)

            async def _call_anthropic(messages: list[dict]) -> str:
                # Anthropic separates system from user messages
                system = next(
                    (m["content"] for m in messages if m.get("role") == "system"), ""
                )
                user_msgs = [m for m in messages if m.get("role") != "system"]
                resp = await client.messages.create(
                    model=model or "claude-haiku-4-5-20251001",
                    max_tokens=512,
                    system=system,
                    messages=user_msgs,
                )
                return resp.content[0].text if resp.content else ""

            return _call_anthropic
        except ImportError:
            return None

    return None


def _run_attribution(response_event: LLMResponseEvent, _bus: Any = None) -> None:
    """Run hallucination detection and chunk attribution for an LLM response."""
    try:
        if _bus is None:
            from vectorlens.session_bus import bus
        else:
            bus = _bus
        from vectorlens.detection.hallucination import HallucinationDetector

        output_text = response_event.output_text
        if not output_text or not output_text.strip():
            return

        # Cap output size before passing to SentenceTransformer to prevent RAM
        # spikes on giant LLM outputs (100k+ tokens). 50k chars covers ~12k tokens
        # which is enough for meaningful attribution on any realistic response.
        _MAX_OUTPUT_CHARS = 50_000
        if len(output_text) > _MAX_OUTPUT_CHARS:
            logger.debug(f"Output text capped at {_MAX_OUTPUT_CHARS} chars for attribution")
            output_text = output_text[:_MAX_OUTPUT_CHARS]

        session = bus.get_session(response_event.session_id)
        if not session:
            return

        # ------------------------------------------------------------------
        # Source resolution: GraphRAG context takes priority over vector queries.
        # GraphRAG pipelines don't use vector DB interceptors — their retrieval
        # goes through the context builder, captured as GraphRAGContextEvent.
        # ------------------------------------------------------------------
        graphrag_ctx = session.graphrag_contexts[-1] if session.graphrag_contexts else None

        if graphrag_ctx and graphrag_ctx.search_type == "global":
            # Global search: attribute via community reports (the novel path)
            _run_graphrag_global_attribution(
                response_event, session, graphrag_ctx, output_text, bus
            )
            return

        # CAG: full corpus in context, no vector DB retrieval
        cag_ctx = session.cag_contexts[-1] if session.cag_contexts else None
        if cag_ctx and cag_ctx.documents:
            _run_cag_attribution(response_event, session, cag_ctx, output_text, bus)
            return

        # Get chunks from the vector query linked to this response, else latest.
        # For GraphRAG local search, text_chunks are used as standard chunks below.
        chunks: list[RetrievedChunk] = []
        if graphrag_ctx and graphrag_ctx.search_type == "local" and graphrag_ctx.text_chunks:
            chunks = graphrag_ctx.text_chunks
        elif session.vector_queries:
            linked_query = None
            for req in session.llm_requests:
                if req.id == response_event.request_id and req.vector_query_id:
                    linked_query = next(
                        (q for q in session.vector_queries if q.id == req.vector_query_id),
                        None,
                    )
                    break
            query = linked_query or session.vector_queries[-1]
            chunks = query.results

        if not chunks:
            logger.debug("No chunks available for attribution, skipping")
            return

        detector = HallucinationDetector()
        output_tokens = detector.detect(output_text, chunks)

        if output_tokens:
            grounded = sum(1 for t in output_tokens if not t.is_hallucinated)
            overall_groundedness = grounded / len(output_tokens)
            hallucinated_count = len(output_tokens) - grounded
        else:
            overall_groundedness = 1.0
            hallucinated_count = 0

        hallucinated_spans: list[tuple[int, int]] = []
        i = 0
        while i < len(output_tokens):
            if output_tokens[i].is_hallucinated:
                start = i
                while i < len(output_tokens) and output_tokens[i].is_hallucinated:
                    i += 1
                hallucinated_spans.append((start, i - 1))
            else:
                i += 1

        # Fallback: use existing token-level attribution scores
        chunk_scores: dict[str, float] = {}
        for token in output_tokens:
            for chunk_id, score in token.chunk_attributions.items():
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)

        for chunk in chunks:
            if chunk.chunk_id in chunk_scores:
                chunk.attribution_score = chunk_scores[chunk.chunk_id]
            if overall_groundedness < 0.5 and chunk.attribution_score < 0.1:
                chunk.caused_hallucination = True

        # CONDITIONAL: skip deep attribution if fully grounded
        if hallucinated_count == 0:
            logger.debug("No hallucinations detected — skipping deep chunk attribution")
            result = AttributionResult(
                session_id=response_event.session_id,
                request_id=response_event.request_id,
                response_id=response_event.id,
                chunks=chunks,
                output_tokens=output_tokens,
                overall_groundedness=overall_groundedness,
                hallucinated_spans=hallucinated_spans,
            )
            bus.record_attribution(result)
            return

        # Deep attribution: try attention for local HF models first
        token_heatmap: list[TokenHeatmapEntry] = []
        hf_model, hf_tokenizer = _try_get_hf_model_and_tokenizer(session, response_event)
        if hf_model is not None and hf_tokenizer is not None:
            try:
                from vectorlens.attribution.attention import AttentionAttributor

                attributor = AttentionAttributor()
                input_text = " ".join(
                    m.get("content", "")
                    for m in (
                        session.llm_requests[-1].messages
                        if session.llm_requests
                        else []
                    )
                    if isinstance(m.get("content"), str)
                )
                if input_text:
                    # Pass a copy to compute_per_token — compute() mutates
                    # chunk.attribution_score in place and we don't want
                    # compute_per_token to receive already-scored chunks.
                    import copy
                    chunks_for_heatmap = copy.deepcopy(chunks)
                    chunks = attributor.compute(
                        hf_model, hf_tokenizer, input_text, output_text, chunks
                    )
                    token_heatmap = attributor.compute_per_token(
                        hf_model, hf_tokenizer, input_text, output_text, chunks_for_heatmap
                    )
                    logger.debug(
                        f"Used attention rollout attribution (HF model), "
                        f"{len(token_heatmap)} token heatmap entries"
                    )
            except Exception as e:
                logger.debug(
                    f"Attention attribution failed, skipping: {e}", exc_info=False
                )
        else:
            # API model (OpenAI/Anthropic/etc) — use LIME perturbation attribution.
            # Costs exactly N_LIME_SAMPLES=7 extra LLM calls, run concurrently.
            llm_request = next(
                (r for r in session.llm_requests if r.id == response_event.request_id),
                session.llm_requests[-1] if session.llm_requests else None,
            )
            if llm_request:
                llm_caller = _make_llm_caller(llm_request.provider, llm_request.model)
                if llm_caller is not None:
                    try:
                        from vectorlens.attribution.perturbation import PerturbationAttributor
                        attributor = PerturbationAttributor(llm_caller)
                        loop = asyncio.new_event_loop()
                        try:
                            chunks = loop.run_until_complete(
                                attributor.compute_lime(
                                    llm_request.messages, chunks, output_text
                                )
                            )
                        finally:
                            loop.close()
                        logger.debug(
                            f"LIME perturbation attribution complete for {len(chunks)} chunks"
                        )
                    except Exception as e:
                        logger.debug(f"LIME attribution failed: {e}", exc_info=False)

        result = AttributionResult(
            session_id=response_event.session_id,
            request_id=response_event.request_id,
            response_id=response_event.id,
            chunks=chunks,
            output_tokens=output_tokens,
            overall_groundedness=overall_groundedness,
            hallucinated_spans=hallucinated_spans,
            token_heatmap=token_heatmap,
        )

        bus.record_attribution(result)
        logger.debug(
            f"Attribution complete: groundedness={overall_groundedness:.2f} "
            f"spans={len(hallucinated_spans)} chunks={len(chunks)}"
        )

    except Exception as e:
        logger.warning(f"Auto-attribution failed: {e}", exc_info=False)


def _run_cag_attribution(
    response_event: LLMResponseEvent,
    session: Any,
    cag_ctx: CAGContextEvent,
    output_text: str,
    bus: Any,
) -> None:
    """Attribution path for Cache-Augmented Generation.

    No retrieval happened — the full document corpus was in the context.
    Uses the same cosine similarity approach as GraphRAG global search:
    hallucinated sentences are compared against all registered documents.
    """
    from vectorlens.attribution.graphrag_attribution import (
        CommunityAttributor,
    )
    from vectorlens.detection.hallucination import HallucinationDetector
    from vectorlens.types import GraphRAGCommunityUnit

    # Convert CAG documents to CommunityUnit-compatible format for reuse of attributor
    pseudo_communities = [
        GraphRAGCommunityUnit(
            unit_id=doc.unit_id,
            community_id=doc.doc_id,
            title=doc.title,
            text=doc.text,
            rank=float(i),
        )
        for i, doc in enumerate(cag_ctx.documents)
        if doc.text
    ]

    pseudo_chunks = [
        RetrievedChunk(chunk_id=u.unit_id, text=u.text, score=1.0)
        for u in pseudo_communities
    ]

    detector = HallucinationDetector()
    output_tokens = detector.detect(output_text, pseudo_chunks)

    grounded = sum(1 for t in output_tokens if not t.is_hallucinated)
    overall_groundedness = grounded / len(output_tokens) if output_tokens else 1.0
    hallucinated_count = len(output_tokens) - grounded

    hallucinated_spans: list[tuple[int, int]] = []
    i = 0
    while i < len(output_tokens):
        if output_tokens[i].is_hallucinated:
            start = i
            while i < len(output_tokens) and output_tokens[i].is_hallucinated:
                i += 1
            hallucinated_spans.append((start, i - 1))
        else:
            i += 1

    if hallucinated_count > 0:
        attributor = CommunityAttributor()
        pseudo_communities = attributor.attribute(output_tokens, pseudo_communities)

    # Serialize as RetrievedChunk with cag_document metadata
    chunks = [
        RetrievedChunk(
            chunk_id=u.unit_id,
            text=u.text,
            score=1.0,
            attribution_score=u.attribution_score,
            caused_hallucination=u.caused_hallucination,
            metadata={
                "type": "cag_document",
                "doc_id": u.community_id,
                "title": u.title,
            },
        )
        for u in pseudo_communities
    ]

    result = AttributionResult(
        session_id=response_event.session_id,
        request_id=response_event.request_id,
        response_id=response_event.id,
        chunks=chunks,
        output_tokens=output_tokens,
        overall_groundedness=overall_groundedness,
        hallucinated_spans=hallucinated_spans,
    )
    bus.record_attribution(result)
    logger.debug(
        f"CAG attribution: groundedness={overall_groundedness:.2f}, "
        f"hallucinated={hallucinated_count}, "
        f"documents={len(pseudo_communities)}"
    )


def _run_graphrag_global_attribution(
    response_event: LLMResponseEvent,
    session: Any,
    graphrag_ctx: GraphRAGContextEvent,
    output_text: str,
    bus: Any,
) -> None:
    """Attribution path for GraphRAG GlobalSearch.

    Uses semantic similarity (Tier 1) to attribute hallucinated sentences to
    community reports — the novel approach that solves the 'no discrete chunk'
    problem.

    Algorithm:
      1. Detect hallucinations against community report texts (not chunks)
      2. For each community report: score = max cosine_sim(community, hallucinated_sentence)
      3. Normalize to [0,1] — high score = community most 'inspired' the hallucination
      4. Record AttributionResult with community units serialized as chunks
    """
    from vectorlens.attribution.graphrag_attribution import (
        CommunityAttributor,
        community_units_to_chunks,
    )
    from vectorlens.detection.hallucination import HallucinationDetector

    community_units = graphrag_ctx.community_units
    if not community_units:
        logger.debug("GraphRAG global context has no community units, skipping")
        return

    # Use community report texts as the "chunks" for hallucination detection.
    # This aligns the detection threshold with what was actually in the prompt.
    pseudo_chunks = [
        RetrievedChunk(
            chunk_id=u.unit_id,
            text=u.text,
            score=u.rank,
        )
        for u in community_units
        if u.text
    ]

    detector = HallucinationDetector()
    output_tokens = detector.detect(output_text, pseudo_chunks)

    grounded = sum(1 for t in output_tokens if not t.is_hallucinated)
    overall_groundedness = grounded / len(output_tokens) if output_tokens else 1.0
    hallucinated_count = len(output_tokens) - grounded

    hallucinated_spans: list[tuple[int, int]] = []
    i = 0
    while i < len(output_tokens):
        if output_tokens[i].is_hallucinated:
            start = i
            while i < len(output_tokens) and output_tokens[i].is_hallucinated:
                i += 1
            hallucinated_spans.append((start, i - 1))
        else:
            i += 1

    if hallucinated_count > 0:
        attributor = CommunityAttributor()
        community_units = attributor.attribute(output_tokens, community_units)

    # Serialize community units as RetrievedChunk for the API response.
    # metadata["type"] = "graphrag_community" lets the dashboard distinguish these.
    chunks = community_units_to_chunks(community_units)

    result = AttributionResult(
        session_id=response_event.session_id,
        request_id=response_event.request_id,
        response_id=response_event.id,
        chunks=chunks,
        output_tokens=output_tokens,
        overall_groundedness=overall_groundedness,
        hallucinated_spans=hallucinated_spans,
    )
    bus.record_attribution(result)
    logger.debug(
        f"GraphRAG global attribution: groundedness={overall_groundedness:.2f}, "
        f"hallucinated={hallucinated_count}, "
        f"communities={len(community_units)}, "
        f"flagged={sum(u.caused_hallucination for u in community_units)}"
    )
