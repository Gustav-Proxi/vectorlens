"""Auto-attribution pipeline.

Subscribes to llm_response events on the bus and automatically runs
hallucination detection + chunk attribution, storing results back to the bus.
Runs in a background thread — no blocking of the main pipeline.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from vectorlens.types import AttributionResult, LLMResponseEvent, OutputToken, RetrievedChunk

logger = logging.getLogger(__name__)

_attribution_lock = threading.Lock()
_installed = False


def setup_auto_attribution() -> None:
    """Subscribe to the bus and run attribution after every LLM response."""
    global _installed
    with _attribution_lock:
        if _installed:
            return
        from vectorlens.session_bus import bus
        bus.subscribe("llm_response", _on_llm_response)
        _installed = True
        logger.info("Auto-attribution pipeline active")


def _on_llm_response(event: LLMResponseEvent) -> None:
    """Triggered after every LLM response — runs attribution in background thread."""
    threading.Thread(
        target=_run_attribution,
        args=(event,),
        daemon=True,
        name="vectorlens-attribution",
    ).start()


def _run_attribution(response_event: LLMResponseEvent) -> None:
    """Run hallucination detection and chunk attribution for an LLM response."""
    try:
        from vectorlens.session_bus import bus
        from vectorlens.detection.hallucination import HallucinationDetector

        output_text = response_event.output_text
        if not output_text or not output_text.strip():
            return

        # Find the session and its chunks
        session = bus.get_session(response_event.session_id)
        if not session:
            return

        # Get chunks from the most recent vector query
        chunks: list[RetrievedChunk] = []
        if session.vector_queries:
            # Use the vector query linked to the corresponding LLM request, else latest
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

        # Run hallucination detection (local embeddings, no API calls)
        detector = HallucinationDetector()
        output_tokens = detector.detect(output_text, chunks)

        # Compute overall groundedness
        if output_tokens:
            grounded = sum(1 for t in output_tokens if not t.is_hallucinated)
            overall_groundedness = grounded / len(output_tokens)
        else:
            overall_groundedness = 1.0

        # Find hallucinated spans (token index ranges)
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

        # Update chunk attribution scores from token-level attributions
        chunk_scores: dict[str, float] = {}
        for token in output_tokens:
            for chunk_id, score in token.chunk_attributions.items():
                chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)

        for chunk in chunks:
            if chunk.chunk_id in chunk_scores:
                chunk.attribution_score = chunk_scores[chunk.chunk_id]
            # Mark chunks that only appear in hallucinated spans
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
            "Attribution complete",
            extra={
                "groundedness": f"{overall_groundedness:.2f}",
                "hallucinated_spans": len(hallucinated_spans),
                "chunks": len(chunks),
            },
        )

    except Exception as e:
        logger.warning(f"Auto-attribution failed: {e}", exc_info=False)
