"""In-process event bus and session manager.

All interceptors publish events here. The server reads from here.
Thread-safe. No external dependencies.

SESSION ISOLATION
-----------------
_active_session_var is a contextvars.ContextVar — not a plain instance
attribute — so each asyncio Task and each thread automatically gets its
own active session. This prevents cross-thread data bleed in concurrent
RAG servers (FastAPI, Flask, ThreadPoolExecutor, etc.):

  Thread A  →  ContextVar["session-A"]  →  User A's chunks + LLM output
  Thread B  →  ContextVar["session-B"]  →  User B's chunks + LLM output

The shared _sessions dict stores all sessions for API retrieval, but
the "which session am I currently writing to?" decision is per-context.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from contextvars import ContextVar
from typing import Callable

from vectorlens.types import (
    AttributionResult,
    CAGContextEvent,
    GraphRAGContextEvent,
    LLMRequestEvent,
    LLMResponseEvent,
    Session,
    VectorQueryEvent,
)

logger = logging.getLogger(__name__)

# Max sessions kept in memory — oldest evicted when exceeded (prevents OOM)
MAX_SESSIONS = 200

# Per-context active session — isolated per asyncio task and per thread.
# Default None means "no session yet for this context; create one on demand".
_active_session_var: ContextVar[str | None] = ContextVar(
    "vectorlens_active_session", default=None
)


class SessionBus:
    """Central event bus — interceptors write, server reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._session_order: list[str] = []  # insertion order for LRU eviction
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Evict oldest sessions (FIFO) when MAX_SESSIONS exceeded. Call under _lock."""
        while len(self._sessions) > MAX_SESSIONS:
            oldest_id = self._session_order.pop(0)
            self._sessions.pop(oldest_id, None)
            logger.debug(f"Evicted session {oldest_id[:8]} (max sessions reached)")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def get_or_create_session(self) -> Session:
        """Return the session for this context, creating one if needed."""
        session_id = _active_session_var.get()
        # Fast path: existing session still present
        if session_id:
            session = self._sessions.get(session_id)
            if session is not None:
                return session
        # Slow path: create a new session for this context
        with self._lock:
            session = Session()
            self._sessions[session.id] = session
            self._session_order.append(session.id)
            _active_session_var.set(session.id)
            self._evict_if_needed()
            return session

    def new_session(self) -> Session:
        """Explicitly start a fresh session for this context."""
        with self._lock:
            session = Session()
            self._sessions[session.id] = session
            self._session_order.append(session.id)
            _active_session_var.set(session.id)
            self._evict_if_needed()
            return session

    def start_conversation(self) -> str:
        """Start a new conversation. Returns conversation_id.

        All sessions created within this conversation share the same
        conversation_id, enabling parent-child trace linking in the dashboard.
        """
        session = self.new_session()
        return session.conversation_id

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())

    def clear_all_sessions(self) -> int:
        """Delete all sessions. Returns count of deleted sessions."""
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            self._session_order.clear()
            _active_session_var.set(None)
            return count

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if found and deleted."""
        with self._lock:
            if session_id not in self._sessions:
                return False
            del self._sessions[session_id]
            try:
                self._session_order.remove(session_id)
            except ValueError:
                pass
            # Clear ContextVar if this was the active session for current context
            if _active_session_var.get() == session_id:
                next_id = self._session_order[-1] if self._session_order else None
                _active_session_var.set(next_id)
            return True

    # Backward-compatible property — some code reads bus._active_session_id
    @property
    def _active_session_id(self) -> str | None:
        return _active_session_var.get()

    @_active_session_id.setter
    def _active_session_id(self, value: str | None) -> None:
        _active_session_var.set(value)

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def _resolve_session(self, session_id: str | None) -> Session:
        """Return the session for session_id if set and valid, else the context session."""
        if session_id:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing
        return self.get_or_create_session()

    def record_vector_query(self, event: VectorQueryEvent) -> None:
        # Hold lock through resolve + append to prevent concurrent appends
        # to the same session list (list resize is not thread-safe under contention)
        with self._lock:
            session = self._resolve_session(event.session_id)
            event.session_id = session.id
            session.vector_queries.append(event)
        self._notify("vector_query", event)

    def record_llm_request(self, event: LLMRequestEvent) -> None:
        with self._lock:
            session = self._resolve_session(event.session_id)
            event.session_id = session.id
            session.llm_requests.append(event)
        self._notify("llm_request", event)

    def record_llm_response(self, event: LLMResponseEvent) -> None:
        with self._lock:
            session = self._resolve_session(event.session_id)
            event.session_id = session.id
            session.llm_responses.append(event)
        self._notify("llm_response", event)

    def record_attribution(self, result: AttributionResult) -> None:
        with self._lock:
            session = self._resolve_session(result.session_id)
            result.session_id = session.id
            session.attributions.append(result)
        self._notify("attribution", result)

    def record_graphrag_context(self, event: GraphRAGContextEvent) -> None:
        with self._lock:
            session = self._resolve_session(event.session_id)
            event.session_id = session.id
            session.graphrag_contexts.append(event)
        self._notify("graphrag_context", event)

    def record_cag_context(self, event: CAGContextEvent) -> None:
        with self._lock:
            session = self._resolve_session(event.session_id)
            event.session_id = session.id
            session.cag_contexts.append(event)
        self._notify("cag_context", event)

    # ------------------------------------------------------------------
    # Pub/sub
    # ------------------------------------------------------------------

    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._listeners[event_type].append(callback)

    def _notify(self, event_type: str, event: object) -> None:
        for cb in self._listeners.get(event_type, []):
            try:
                cb(event)
            except Exception:
                logger.debug(
                    f"Subscriber callback failed for {event_type}", exc_info=True
                )


# Global singleton — interceptors import this
bus = SessionBus()
