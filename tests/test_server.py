"""Tests for FastAPI server and REST API endpoints."""
import pytest
from fastapi.testclient import TestClient

from vectorlens.server.app import app
from vectorlens.session_bus import SessionBus, bus
from vectorlens.types import (
    LLMRequestEvent,
    LLMResponseEvent,
    Session,
    VectorQueryEvent,
    RetrievedChunk,
    OutputToken,
    AttributionResult,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def fresh_bus():
    """Provide fresh bus state for tests."""
    bus._sessions.clear()
    bus._session_order.clear()
    bus._active_session_id = None
    yield bus
    bus._sessions.clear()
    bus._session_order.clear()
    bus._active_session_id = None


# ============================================================================
# Status Endpoint Tests
# ============================================================================


def test_status_endpoint(client, fresh_bus):
    """Test GET /api/status returns server status."""
    response = client.get("/api/status")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "ok"
    assert "interceptors" in data
    assert "active_session" in data
    assert isinstance(data["interceptors"], list)


def test_status_active_session_is_none_initially(client, fresh_bus):
    """Test that active_session is None when no session created."""
    response = client.get("/api/status")
    data = response.json()
    assert data["active_session"] is None


def test_status_active_session_after_creation(client, fresh_bus):
    """Test that active_session is set after creating session."""
    # Create session
    create_resp = client.post("/api/sessions/new")
    session_id = create_resp.json()["id"]

    # Check status
    status_resp = client.get("/api/status")
    data = status_resp.json()
    assert data["active_session"] == session_id


# ============================================================================
# Session List Endpoint Tests
# ============================================================================


def test_sessions_list_empty(client, fresh_bus):
    """Test GET /api/sessions returns empty list initially."""
    response = client.get("/api/sessions")
    assert response.status_code == 200
    assert response.json() == []


def test_sessions_list_after_creation(client, fresh_bus):
    """Test sessions list includes created sessions."""
    # Create multiple sessions
    for _ in range(3):
        client.post("/api/sessions/new")

    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 3

    # Each should have required fields
    for session in sessions:
        assert "id" in session
        assert "created_at" in session
        assert "vector_queries_count" in session
        assert "llm_requests_count" in session


def test_sessions_list_respects_counts(client, fresh_bus):
    """Test that session counts are accurate."""
    # Create session
    client.post("/api/sessions/new")

    # Add some events to active session (via bus)
    session = bus.get_or_create_session()
    session.vector_queries.append(VectorQueryEvent())
    session.vector_queries.append(VectorQueryEvent())
    session.llm_requests.append(LLMRequestEvent())

    response = client.get("/api/sessions")
    sessions = response.json()
    assert sessions[0]["vector_queries_count"] == 2
    assert sessions[0]["llm_requests_count"] == 1
    assert sessions[0]["llm_responses_count"] == 0


# ============================================================================
# Create Session Endpoint Tests
# ============================================================================


def test_create_new_session(client, fresh_bus):
    """Test POST /api/sessions/new creates session."""
    response = client.post("/api/sessions/new")
    assert response.status_code == 200

    data = response.json()
    assert "id" in data
    assert "created_at" in data
    assert isinstance(data["id"], str)
    assert isinstance(data["created_at"], (int, float))


def test_create_session_sets_active(client, fresh_bus):
    """Test that created session becomes active."""
    response = client.post("/api/sessions/new")
    session_id = response.json()["id"]

    # Check status reflects active session
    status = client.get("/api/status").json()
    assert status["active_session"] == session_id


def test_create_multiple_sessions(client, fresh_bus):
    """Test creating multiple sessions."""
    ids = []
    for _ in range(3):
        resp = client.post("/api/sessions/new")
        ids.append(resp.json()["id"])

    # All should be unique
    assert len(set(ids)) == 3

    # Latest should be active
    status = client.get("/api/status").json()
    assert status["active_session"] == ids[-1]


# ============================================================================
# Get Session Endpoint Tests
# ============================================================================


def test_get_session_by_id(client, fresh_bus):
    """Test GET /api/sessions/{session_id}."""
    # Create session
    create_resp = client.post("/api/sessions/new")
    session_id = create_resp.json()["id"]

    # Get it back
    get_resp = client.get(f"/api/sessions/{session_id}")
    assert get_resp.status_code == 200

    data = get_resp.json()
    assert data["id"] == session_id
    assert "created_at" in data
    assert "vector_queries" in data
    assert "llm_requests" in data
    assert "llm_responses" in data
    assert "attributions" in data


def test_get_nonexistent_session(client, fresh_bus):
    """Test getting nonexistent session returns 404."""
    response = client.get("/api/sessions/nonexistent-id-12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_session_with_events(client, fresh_bus):
    """Test getting session with events populated."""
    # Create session
    client.post("/api/sessions/new")
    session = bus.get_or_create_session()

    # Add events
    session.vector_queries.append(VectorQueryEvent(
        db_type="chroma",
        collection="test",
        query_text="test",
        top_k=5,
    ))
    session.llm_requests.append(LLMRequestEvent(
        provider="openai",
        model="gpt-4",
    ))
    session.llm_responses.append(LLMResponseEvent(
        request_id="req1",
        output_text="Response text",
    ))

    # Get session
    response = client.get(f"/api/sessions/{session.id}")
    data = response.json()

    assert len(data["vector_queries"]) == 1
    assert len(data["llm_requests"]) == 1
    assert len(data["llm_responses"]) == 1
    assert data["llm_responses"][0]["output_text"] == "Response text"


# ============================================================================
# Delete Session Endpoint Tests
# ============================================================================


def test_delete_session(client, fresh_bus):
    """Test DELETE /api/sessions/{session_id}."""
    # Create session
    create_resp = client.post("/api/sessions/new")
    session_id = create_resp.json()["id"]

    # Delete it
    del_resp = client.delete(f"/api/sessions/{session_id}")
    assert del_resp.status_code == 204

    # Should be gone
    get_resp = client.get(f"/api/sessions/{session_id}")
    assert get_resp.status_code == 404


def test_delete_nonexistent_session(client, fresh_bus):
    """Test deleting nonexistent session returns 404."""
    response = client.delete("/api/sessions/nonexistent-12345")
    assert response.status_code == 404


def test_delete_clears_active_session(client, fresh_bus):
    """Test deleting active session clears active_session_id."""
    # Create single session
    resp = client.post("/api/sessions/new")
    session_id = resp.json()["id"]

    # Verify it's active
    status = client.get("/api/status").json()
    assert status["active_session"] == session_id

    # Delete it
    client.delete(f"/api/sessions/{session_id}")

    # Should clear active session
    status = client.get("/api/status").json()
    assert status["active_session"] is None


def test_delete_one_session_keeps_others(client, fresh_bus):
    """Test that deleting one session doesn't affect others."""
    # Create two sessions
    id1 = client.post("/api/sessions/new").json()["id"]
    id2 = client.post("/api/sessions/new").json()["id"]

    # Delete first
    client.delete(f"/api/sessions/{id1}")

    # Second should still exist
    response = client.get(f"/api/sessions/{id2}")
    assert response.status_code == 200
    assert response.json()["id"] == id2


# ============================================================================
# Get Session Attributions Endpoint Tests
# ============================================================================


def test_session_attributions_empty(client, fresh_bus):
    """Test GET /api/sessions/{session_id}/attributions for empty session."""
    # Create session
    resp = client.post("/api/sessions/new")
    session_id = resp.json()["id"]

    # Get attributions
    attr_resp = client.get(f"/api/sessions/{session_id}/attributions")
    assert attr_resp.status_code == 200
    assert attr_resp.json() == []


def test_session_attributions_with_data(client, fresh_bus):
    """Test getting attributions for session with data."""
    # Create session
    client.post("/api/sessions/new")
    session = bus.get_or_create_session()

    # Add attribution
    attribution = AttributionResult(
        request_id="req1",
        response_id="resp1",
        overall_groundedness=0.85,
    )
    session.attributions.append(attribution)

    # Get attributions
    response = client.get(f"/api/sessions/{session.id}/attributions")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 1
    assert data[0]["overall_groundedness"] == 0.85


def test_session_attributions_nonexistent_session(client, fresh_bus):
    """Test getting attributions for nonexistent session."""
    response = client.get("/api/sessions/nonexistent/attributions")
    assert response.status_code == 404


def test_session_attributions_with_chunks(client, fresh_bus):
    """Test attribution data includes chunk information."""
    client.post("/api/sessions/new")
    session = bus.get_or_create_session()

    chunk = RetrievedChunk(
        chunk_id="chunk1",
        text="Important fact",
        score=0.95,
        attribution_score=0.8,
    )

    attribution = AttributionResult(
        request_id="req1",
        response_id="resp1",
        chunks=[chunk],
        overall_groundedness=0.9,
    )
    session.attributions.append(attribution)

    response = client.get(f"/api/sessions/{session.id}/attributions")
    data = response.json()

    assert len(data[0]["chunks"]) == 1
    assert data[0]["chunks"][0]["chunk_id"] == "chunk1"
    assert data[0]["chunks"][0]["attribution_score"] == 0.8


# ============================================================================
# Response Format Tests
# ============================================================================


def test_response_has_correct_schema(client, fresh_bus):
    """Test that responses follow Pydantic schema."""
    # Create session with data
    client.post("/api/sessions/new")
    session = bus.get_or_create_session()

    # Add various events
    chunk = RetrievedChunk(
        chunk_id="c1",
        text="text",
        score=0.9,
        metadata={"source": "doc.txt"},
    )
    session.vector_queries.append(VectorQueryEvent(
        db_type="chroma",
        collection="docs",
        query_text="query",
        results=[chunk],
    ))

    token = OutputToken(
        text="Output token",
        position=0,
        is_hallucinated=False,
        hallucination_score=0.1,
    )
    session.llm_responses.append(LLMResponseEvent(
        request_id="r1",
        output_text="Response",
        output_tokens=[token],
    ))

    # Get session detail
    response = client.get(f"/api/sessions/{session.id}")
    data = response.json()

    # Check nested schema
    assert len(data["vector_queries"]) == 1
    vq = data["vector_queries"][0]
    assert vq["db_type"] == "chroma"
    assert len(vq["results"]) == 1

    assert len(data["llm_responses"]) == 1
    resp = data["llm_responses"][0]
    assert len(resp["output_tokens"]) == 1
    assert resp["output_tokens"][0]["is_hallucinated"] is False


def test_timestamp_is_float(client, fresh_bus):
    """Test that timestamps are returned as floats."""
    resp = client.post("/api/sessions/new")
    session_data = resp.json()

    assert isinstance(session_data["created_at"], (int, float))


def test_metadata_preserved(client, fresh_bus):
    """Test that chunk metadata is preserved in responses."""
    client.post("/api/sessions/new")
    session = bus.get_or_create_session()

    chunk = RetrievedChunk(
        chunk_id="c1",
        text="text",
        score=0.9,
        metadata={
            "source": "document.pdf",
            "page": 5,
            "section": "Introduction",
        },
    )
    session.vector_queries.append(VectorQueryEvent(
        results=[chunk],
    ))

    response = client.get(f"/api/sessions/{session.id}")
    data = response.json()

    retrieved = data["vector_queries"][0]["results"][0]
    assert retrieved["metadata"]["source"] == "document.pdf"
    assert retrieved["metadata"]["page"] == 5


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_get_sessions_with_large_dataset(client, fresh_bus):
    """Test listing sessions with many sessions."""
    # Create many sessions
    for _ in range(100):
        client.post("/api/sessions/new")

    response = client.get("/api/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert len(sessions) == 100


def test_session_with_empty_events_list(client, fresh_bus):
    """Test session with empty event lists."""
    resp = client.post("/api/sessions/new")
    session_id = resp.json()["id"]

    get_resp = client.get(f"/api/sessions/{session_id}")
    data = get_resp.json()

    assert data["vector_queries"] == []
    assert data["llm_requests"] == []
    assert data["llm_responses"] == []
    assert data["attributions"] == []


def test_concurrent_session_creation(client, fresh_bus):
    """Test that concurrent requests don't cause issues."""
    import threading

    session_ids = []
    lock = threading.Lock()

    def create_session():
        resp = client.post("/api/sessions/new")
        with lock:
            session_ids.append(resp.json()["id"])

    threads = [threading.Thread(target=create_session) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have created all sessions without conflicts
    assert len(session_ids) == 10
    assert len(set(session_ids)) == 10  # All unique
