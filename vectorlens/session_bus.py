"""In-process event bus and session manager.

All interceptors publish events here. The server reads from here.
Thread-safe. No external dependencies.
"""
from __future__ import annotations

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


class SessionBus:
    """Central event bus — interceptors write, server reads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._active_session_id: str | None = None
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    def get_or_create_session(self) -> Session:
        with self._lock:
            if self._active_session_id and self._active_session_id in self._sessions:
                return self._sessions[self._active_session_id]
            session = Session()
            self._sessions[session.id] = session
            self._active_session_id = session.id
            return session

    def new_session(self) -> Session:
        with self._lock:
            session = Session()
            self._sessions[session.id] = session
            self._active_session_id = session.id
            return session

    def get_session(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def all_sessions(self) -> list[Session]:
        return list(self._sessions.values())

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
                pass


# Global singleton — interceptors import this
bus = SessionBus()
