"""In-process event bus and session manager.

All interceptors publish events here. The server reads from here.
Thread-safe. No external dependencies.
"""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Callable

from vectorlens.types import (
    AttributionResult,
    LLMRequestEvent,
    LLMResponseEvent,
    Session,
    VectorQueryEvent,
)

logger = logging.getLogger(__name__)

# Max sessions kept in memory — oldest evicted when exceeded (prevents OOM)
MAX_SESSIONS = 200


class SessionBus:
    """Central event bus — interceptors write, server reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._session_order: list[str] = []  # insertion order for LRU eviction
        self._active_session_id: str | None = None
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    def _evict_if_needed(self) -> None:
        """Evict oldest sessions when MAX_SESSIONS is exceeded. Call under _lock."""
        while len(self._sessions) > MAX_SESSIONS:
            oldest_id = self._session_order.pop(0)
            # Don't evict the active session
            if oldest_id == self._active_session_id:
                self._session_order.append(oldest_id)
                break
            self._sessions.pop(oldest_id, None)
            logger.debug(f"Evicted session {oldest_id[:8]} (max sessions reached)")

    def get_or_create_session(self) -> Session:
        with self._lock:
            if self._active_session_id and self._active_session_id in self._sessions:
                return self._sessions[self._active_session_id]
            session = Session()
            self._sessions[session.id] = session
            self._session_order.append(session.id)
            self._active_session_id = session.id
            self._evict_if_needed()
            return session

    def new_session(self) -> Session:
        with self._lock:
            session = Session()
            self._sessions[session.id] = session
            self._session_order.append(session.id)
            self._active_session_id = session.id
            self._evict_if_needed()
            return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())

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
            if self._active_session_id == session_id:
                self._active_session_id = self._session_order[-1] if self._session_order else None
            return True

    def record_vector_query(self, event: VectorQueryEvent) -> None:
        session = self.get_or_create_session()
        event.session_id = session.id
        with self._lock:
            session.vector_queries.append(event)
        self._notify("vector_query", event)

    def record_llm_request(self, event: LLMRequestEvent) -> None:
        session = self.get_or_create_session()
        event.session_id = session.id
        with self._lock:
            session.llm_requests.append(event)
        self._notify("llm_request", event)

    def record_llm_response(self, event: LLMResponseEvent) -> None:
        session = self.get_or_create_session()
        event.session_id = session.id
        with self._lock:
            session.llm_responses.append(event)
        self._notify("llm_response", event)

    def record_attribution(self, result: AttributionResult) -> None:
        session = self.get_or_create_session()
        result.session_id = session.id
        with self._lock:
            session.attributions.append(result)
        self._notify("attribution", result)

    def subscribe(self, event_type: str, callback: Callable) -> None:
        self._listeners[event_type].append(callback)

    def _notify(self, event_type: str, event: object) -> None:
        for cb in self._listeners.get(event_type, []):
            try:
                cb(event)
            except Exception:
                logger.debug(f"Subscriber callback failed for {event_type}", exc_info=True)


# Global singleton — interceptors import this
bus = SessionBus()
