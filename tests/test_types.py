"""Tests for core types and session bus."""
import time
import pytest
from vectorlens.types import (
    Session,
    RetrievedChunk,
    OutputToken,
    LLMRequestEvent,
    LLMResponseEvent,
    VectorQueryEvent,
    AttributionResult,
)
from vectorlens.session_bus import SessionBus


def test_session_defaults():
    s = Session()
    assert s.id
    assert s.created_at <= time.time()
    assert s.vector_queries == []
    assert s.llm_requests == []
    assert s.llm_responses == []
    assert s.attributions == []


def test_retrieved_chunk_defaults():
    c = RetrievedChunk(chunk_id="c1", text="hello", score=0.9)
    assert c.attribution_score == 0.0
    assert not c.caused_hallucination


def test_output_token_defaults():
    t = OutputToken(text="Paris", position=0)
    assert not t.is_hallucinated
    assert t.hallucination_score == 0.0
    assert t.chunk_attributions == {}


class TestSessionBus:
    def setup_method(self):
        self.bus = SessionBus()

    def test_creates_session_on_demand(self):
        s = self.bus.get_or_create_session()
        assert s.id
        assert len(self.bus.all_sessions()) == 1

    def test_reuses_active_session(self):
        s1 = self.bus.get_or_create_session()
        s2 = self.bus.get_or_create_session()
        assert s1.id == s2.id

    def test_new_session_replaces_active(self):
        s1 = self.bus.new_session()
        s2 = self.bus.new_session()
        assert s1.id != s2.id
        assert len(self.bus.all_sessions()) == 2

    def test_record_vector_query(self):
        event = VectorQueryEvent(db_type="chroma", query_text="test")
        self.bus.record_vector_query(event)
        session = self.bus.get_or_create_session()
        assert len(session.vector_queries) == 1
        assert session.vector_queries[0].query_text == "test"
        assert session.vector_queries[0].session_id == session.id

    def test_record_llm_request(self):
        event = LLMRequestEvent(provider="openai", model="gpt-4o")
        self.bus.record_llm_request(event)
        session = self.bus.get_or_create_session()
        assert len(session.llm_requests) == 1

    def test_record_llm_response(self):
        event = LLMResponseEvent(output_text="Paris is the capital.")
        self.bus.record_llm_response(event)
        session = self.bus.get_or_create_session()
        assert len(session.llm_responses) == 1

    def test_subscribe_and_notify(self):
        received = []
        self.bus.subscribe("vector_query", received.append)
        event = VectorQueryEvent(query_text="test")
        self.bus.record_vector_query(event)
        assert len(received) == 1
        assert received[0].query_text == "test"

    def test_get_session_by_id(self):
        s = self.bus.new_session()
        found = self.bus.get_session(s.id)
        assert found is not None
        assert found.id == s.id

    def test_get_missing_session(self):
        assert self.bus.get_session("nonexistent") is None
