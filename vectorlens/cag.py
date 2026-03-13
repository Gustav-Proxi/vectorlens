"""Cache-Augmented Generation (CAG) support for VectorLens.

CAG loads the full document corpus into the LLM context window instead of
retrieving relevant chunks at query time. This works with large-context models
(Gemini 1.5, Claude 3.5, GPT-4o) and avoids retrieval latency entirely.

VectorLens treats each document as an attribution unit — the same cosine
similarity approach used for GraphRAG global search. When the LLM hallucinates,
VectorLens finds which document was closest to the hallucinated content.

Usage:
    import vectorlens

    docs = [
        {"id": "q4-report", "title": "Q4 2025 Report", "text": "Revenue was..."},
        "Raw string documents are also supported.",
    ]

    with vectorlens.cag_session(docs):
        prompt = build_cag_prompt(docs, user_query)
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
    # VectorLens dashboard shows which document caused any hallucination
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any

from vectorlens.types import CAGContextEvent, CAGDocumentUnit

logger = logging.getLogger(__name__)

# Per-context CAG corpus — isolated per asyncio task and thread
_active_cag_var: ContextVar[str | None] = ContextVar(
    "vectorlens_active_cag", default=None
)

# Registry of active CAG events by ID (to survive context boundaries)
_cag_registry: dict[str, CAGContextEvent] = {}


def _register_event(event: CAGContextEvent) -> None:
    _cag_registry[event.id] = event
    _active_cag_var.set(event.id)


def _unregister_event(event_id: str) -> None:
    _cag_registry.pop(event_id, None)
    if _active_cag_var.get() == event_id:
        _active_cag_var.set(None)


def get_active_cag_event() -> CAGContextEvent | None:
    """Return the CAG context event for the current context, if any."""
    event_id = _active_cag_var.get()
    if event_id:
        return _cag_registry.get(event_id)
    return None


def _parse_documents(documents: list[Any]) -> list[CAGDocumentUnit]:
    """Convert user-provided documents to CAGDocumentUnit list.

    Accepts:
    - str: raw document text
    - dict with 'text' key (and optional 'id', 'title')
    - object with .text attribute
    """
    units = []
    for i, doc in enumerate(documents):
        if isinstance(doc, str):
            units.append(CAGDocumentUnit(
                doc_id=str(i),
                title=f"Document {i + 1}",
                text=doc.strip(),
            ))
        elif isinstance(doc, dict):
            text = doc.get("text") or doc.get("content") or doc.get("body") or ""
            units.append(CAGDocumentUnit(
                doc_id=str(doc.get("id", i)),
                title=str(doc.get("title", f"Document {i + 1}")),
                text=text.strip(),
            ))
        else:
            text = getattr(doc, "text", None) or getattr(doc, "content", None) or str(doc)
            units.append(CAGDocumentUnit(
                doc_id=str(getattr(doc, "id", i)),
                title=str(getattr(doc, "title", f"Document {i + 1}")),
                text=str(text).strip(),
            ))
    return [u for u in units if u.text]


class CAGSession:
    """Context manager that registers a document corpus for CAG attribution.

    While active, VectorLens will attribute LLM outputs to documents in
    this corpus rather than to vector DB retrieved chunks.

    Example:
        with vectorlens.cag_session(documents) as session:
            response = llm.complete(build_full_context_prompt(documents, query))
    """

    def __init__(self, documents: list[Any]) -> None:
        self._documents = documents
        self._event: CAGContextEvent | None = None

    def __enter__(self) -> CAGSession:
        from vectorlens.session_bus import bus

        doc_units = _parse_documents(self._documents)
        if not doc_units:
            logger.warning("CAGSession: no valid documents provided — attribution will be skipped")
            return self

        self._event = CAGContextEvent(documents=doc_units)
        _register_event(self._event)
        bus.record_cag_context(self._event)
        logger.debug(f"CAGSession started: {len(doc_units)} documents registered")
        return self

    def __exit__(self, *args: Any) -> None:
        if self._event is not None:
            _unregister_event(self._event.id)
            logger.debug("CAGSession ended")

    # Support async context manager too
    async def __aenter__(self) -> CAGSession:
        return self.__enter__()

    async def __aexit__(self, *args: Any) -> None:
        self.__exit__()

    @property
    def documents(self) -> list[CAGDocumentUnit]:
        return self._event.documents if self._event else []
