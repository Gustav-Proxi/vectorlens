"""Auto-attribution pipeline.

Subscribes to llm_response events on the bus and automatically runs
hallucination detection + chunk attribution, storing results back to the bus.
Uses a bounded ThreadPoolExecutor — prevents thread explosion under load.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from vectorlens.types import AttributionResult, LLMResponseEvent, RetrievedChunk

logger = logging.getLogger(__name__)

_attribution_lock = threading.Lock()
_installed = False

# Bounded pool: max 3 concurrent attribution jobs (prevents thread exhaustion)
_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="vectorlens-attr")


def setup_auto_attribution() -> None:
    """Subscribe to the bus and run attribution after every LLM response."""
    global _installed
    with _attribution_lock:
        if _installed:
            return
        from vectorlens.session_bus import bus
        bus.subscribe("llm_response", _on_llm_response)
        _installed = True
        logger.info("Auto-attribution pipeline active (max_workers=3)")


def _on_llm_response(event: LLMResponseEvent) -> None:
    """Triggered after every LLM response — submits to bounded pool."""
    _executor.submit(_run_attribution, event)


def _run_attribution(response_event: LLMResponseEvent) -> None:
    """Run hallucination detection and chunk attribution for an LLM response."""
    try:
        from vectorlens.session_bus import bus
        from vectorlens.detection.hallucination import HallucinationDetector

        output_text = response_event.output_text
        if not output_text or not output_text.strip():
            return

        session = bus.get_session(response_event.session_id)
        if not session:
            return

        # Get chunks from the vector query linked to this response, else latest
        chunks: list[RetrievedChunk] = []
        if session.vector_queries:
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
        else:
            overall_groundedness = 1.0

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

        chunk_scores: dict[str, float] = {}
        for token in output_tokens:
            for chunk_id, score in token.chunk_attributions.items():
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)

        for chunk in chunks:
            if chunk.chunk_id in chunk_scores:
                chunk.attribution_score = chunk_scores[chunk.chunk_id]
            if overall_groundedness < 0.5 and chunk.attribution_score < 0.1:
                chunk.caused_hallucination = True

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
            f"Attribution complete: groundedness={overall_groundedness:.2f} "
            f"spans={len(hallucinated_spans)} chunks={len(chunks)}"
        )

    except Exception as e:
        logger.warning(f"Auto-attribution failed: {e}", exc_info=False)
