"""Integration tests for the VectorLens RAG tracing pipeline.

Full end-to-end tests that exercise the complete pipeline without mocking,
using real embeddings for hallucination detection.

These tests are slower due to model loading and are marked with @pytest.mark.integration
so they can be skipped with: pytest -m "not integration"
"""
import time

import pytest

from vectorlens.pipeline import setup_auto_attribution, _on_llm_response, _run_attribution
from vectorlens.session_bus import SessionBus
from vectorlens.types import (
    LLMRequestEvent,
    LLMResponseEvent,
    RetrievedChunk,
    VectorQueryEvent,
)


@pytest.mark.integration
def test_full_rag_tracing_pipeline():
    """Full end-to-end integration: vector query → LLM response → attribution.

    Simulates a complete RAG pipeline:
    1. VectorQuery recorded (simulates ChromaDB/Pinecone/etc)
    2. LLMResponse recorded (simulates Anthropic/OpenAI/etc)
    3. Auto-attribution runs with REAL HallucinationDetector
    4. Attribution result is stored in session

    Uses real sentence-transformers model, so this is slower.
    """
    # Use a fresh bus for this integration test
    bus = SessionBus()

    # Create a new session
    session = bus.new_session()

    # Step 1: Record a vector query with retrieved chunks
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Paris is the capital of France.",
            score=0.95,
            metadata={"source": "wiki/paris"},
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="The Eiffel Tower is located in Paris, France.",
            score=0.85,
            metadata={"source": "wiki/eiffel_tower"},
        ),
    ]
    vq = VectorQueryEvent(
        db_type="test_chroma",
        collection="documents",
        query_text="Paris facts",
        top_k=2,
        results=chunks,
    )
    bus.record_vector_query(vq)

    # Verify vector query was recorded
    session = bus.get_session(session.id)
    assert len(session.vector_queries) == 1
    assert session.vector_queries[0].query_text == "Paris facts"
    assert len(session.vector_queries[0].results) == 2

    # Step 2: Record an LLM request
    llm_req = LLMRequestEvent(
        provider="anthropic",
        model="claude-3-sonnet",
        system_prompt="You are a helpful assistant.",
        messages=[
            {
                "role": "user",
                "content": "Tell me about Paris based on the retrieved documents.",
            }
        ],
        temperature=0.7,
        max_tokens=512,
        vector_query_id=vq.id,
    )
    bus.record_llm_request(llm_req)

    session = bus.get_session(session.id)
    assert len(session.llm_requests) == 1

    # Step 3: Record LLM response (mix of grounded and hallucinated content)
    # "Paris is the capital of France." — grounded
    # "The Eiffel Tower is in Paris." — grounded
    # "The Colosseum is in Paris." — HALLUCINATED (Colosseum is in Rome)
    response_text = (
        "Paris is the capital of France. "
        "The Eiffel Tower is located in Paris, France. "
        "The Colosseum is in Paris."
    )

    resp = LLMResponseEvent(
        output_text=response_text,
        request_id=llm_req.id,
        latency_ms=1250.5,
        prompt_tokens=42,
        completion_tokens=51,
        cost_usd=0.00123,
    )
    bus.record_llm_response(resp)

    session = bus.get_session(session.id)
    assert len(session.llm_responses) == 1
    assert session.llm_responses[0].output_text == response_text

    # Step 4: Run attribution directly (synchronous, uses local bus)
    _run_attribution(resp, _bus=bus)

    # Step 5: Verify attribution was computed
    session = bus.get_session(session.id)
    assert len(session.attributions) == 1, "Attribution should be recorded"

    attr = session.attributions[0]

    # Verify attribution result structure
    assert attr.session_id == session.id
    assert attr.request_id == llm_req.id
    assert attr.response_id == resp.id
    assert len(attr.chunks) == 2, "Should have 2 chunks from vector query"
    assert len(attr.output_tokens) > 0, "Should have output tokens"

    # Verify groundedness
    # With real model: 2 sentences grounded, 1 hallucinated ≈ 0.67
    assert 0.0 <= attr.overall_groundedness <= 1.0
    # Due to semantic similarity, "The Colosseum is in Paris" might score low
    # but sentences about Paris capital and Eiffel Tower should be grounded
    assert attr.overall_groundedness >= 0.4, "At least 40% should be grounded"

    # Verify hallucinated spans exist
    assert isinstance(attr.hallucinated_spans, list)

    # Verify output tokens have the expected structure
    for token in attr.output_tokens:
        assert token.text is not None
        assert token.position >= 0
        assert isinstance(token.is_hallucinated, bool)
        assert 0.0 <= token.hallucination_score <= 1.0
        assert isinstance(token.chunk_attributions, dict)


@pytest.mark.integration
def test_multiple_vector_queries_in_session():
    """Test attribution when multiple vector queries exist in the session.

    Verifies that the pipeline correctly selects the right vector query's
    chunks for attribution when multiple queries have been performed.
    """
    bus = SessionBus()
    session = bus.new_session()

    # First vector query
    chunks1 = [
        RetrievedChunk(
            chunk_id="c1",
            text="The Great Wall of China is very long.",
            score=0.92,
        ),
    ]
    vq1 = VectorQueryEvent(
        db_type="test",
        query_text="Great Wall",
        results=chunks1,
    )
    bus.record_vector_query(vq1)

    # Second vector query (more recent)
    chunks2 = [
        RetrievedChunk(
            chunk_id="c2",
            text="The Statue of Liberty is in New York.",
            score=0.90,
        ),
    ]
    vq2 = VectorQueryEvent(
        db_type="test",
        query_text="Statue of Liberty",
        results=chunks2,
    )
    bus.record_vector_query(vq2)

    # LLM request linked to second query
    llm_req = LLMRequestEvent(
        model="test",
        vector_query_id=vq2.id,
    )
    bus.record_llm_request(llm_req)

    resp = LLMResponseEvent(
        output_text="The Statue of Liberty is in New York.",
        request_id=llm_req.id,
    )
    bus.record_llm_response(resp)
    _run_attribution(resp, _bus=bus)

    session = bus.get_session(session.id)
    assert len(session.attributions) == 1

    attr = session.attributions[0]
    # Should use chunks from vq2, not vq1
    chunk_ids = {c.chunk_id for c in attr.chunks}
    assert "c2" in chunk_ids
    assert "c1" not in chunk_ids


@pytest.mark.integration
def test_attribution_with_no_vector_query():
    """Test that attribution gracefully handles LLM response with no vector query."""
    bus = SessionBus()
    session = bus.new_session()

    # No vector query recorded — just an LLM response
    llm_req = LLMRequestEvent(model="test")
    bus.record_llm_request(llm_req)


    resp = LLMResponseEvent(
        output_text="Some output without any context.",
        request_id=llm_req.id,
    )
    bus.record_llm_response(resp)

    time.sleep(1)

    session = bus.get_session(session.id)
    # Attribution should not be recorded (no chunks to attribute)
    assert len(session.attributions) == 0


@pytest.mark.integration
def test_real_semantic_similarity_detection():
    """Test hallucination detection using real semantic similarity.

    Verifies that sentences with high semantic similarity to chunks
    are marked as grounded, while dissimilar sentences are hallucinated.
    """
    bus = SessionBus()
    session = bus.new_session()

    # Retrieve chunks about specific facts
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="London is the capital of England.",
            score=0.96,
        ),
    ]
    vq = VectorQueryEvent(
        db_type="test",
        query_text="London capital",
        results=chunks,
    )
    bus.record_vector_query(vq)

    llm_req = LLMRequestEvent(model="test", vector_query_id=vq.id)
    bus.record_llm_request(llm_req)


    # Output with semantic similarity:
    # "London is the capital of England." — exact match, grounded
    # "London serves as the capital." — similar meaning, should be grounded
    # "Giraffes can jump very high." — completely unrelated, hallucinated
    response_text = (
        "London is the capital of England. "
        "London serves as the capital. "
        "Giraffes can jump very high."
    )

    resp = LLMResponseEvent(
        output_text=response_text,
        request_id=llm_req.id,
    )
    bus.record_llm_response(resp)

    _run_attribution(resp, _bus=bus)

    session = bus.get_session(session.id)
    assert len(session.attributions) == 1

    attr = session.attributions[0]
    assert len(attr.output_tokens) == 3

    # First sentence should be grounded
    token1 = attr.output_tokens[0]
    assert not token1.is_hallucinated, "First token should be grounded"

    # Third sentence should be hallucinated (unrelated)
    token3 = attr.output_tokens[2]
    assert token3.is_hallucinated, "Third token should be hallucinated"

    # Overall groundedness should reflect this mix
    assert 0.3 <= attr.overall_groundedness <= 0.7


@pytest.mark.integration
def test_chunk_attribution_scores_with_real_detection():
    """Test that chunk attribution scores are properly computed with real detection."""
    bus = SessionBus()
    session = bus.new_session()

    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Machine learning powers modern AI systems.",
            score=0.93,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Neural networks are inspired by biological neurons.",
            score=0.88,
        ),
    ]
    vq = VectorQueryEvent(
        db_type="test",
        query_text="machine learning",
        results=chunks,
    )
    bus.record_vector_query(vq)

    llm_req = LLMRequestEvent(model="test", vector_query_id=vq.id)
    bus.record_llm_request(llm_req)


    # Output mentioning both concepts
    response_text = (
        "Machine learning is a fundamental technology in AI. "
        "Neural networks are the backbone of deep learning."
    )

    resp = LLMResponseEvent(
        output_text=response_text,
        request_id=llm_req.id,
    )
    bus.record_llm_response(resp)

    _run_attribution(resp, _bus=bus)

    session = bus.get_session(session.id)
    attr = session.attributions[0]

    # Chunks should have attribution scores updated
    for chunk in attr.chunks:
        # Each chunk should have some attribution from the tokens
        assert 0.0 <= chunk.attribution_score <= 1.0

    # At least one chunk should have a positive attribution
    chunk_scores = [c.attribution_score for c in attr.chunks]
    assert max(chunk_scores) > 0.0


@pytest.mark.integration
def test_session_isolation():
    """Test that multiple sessions don't interfere with each other."""
    bus = SessionBus()

    # Create first session
    session1 = bus.new_session()
    chunks1 = [
        RetrievedChunk(
            chunk_id="c1",
            text="Paris is in France.",
            score=0.95,
        ),
    ]
    vq1 = VectorQueryEvent(
        db_type="test",
        query_text="Paris",
        results=chunks1,
    )
    bus.record_vector_query(vq1)

    # Create second session
    session2 = bus.new_session()
    chunks2 = [
        RetrievedChunk(
            chunk_id="c2",
            text="London is in England.",
            score=0.95,
        ),
    ]
    vq2 = VectorQueryEvent(
        db_type="test",
        query_text="London",
        results=chunks2,
    )
    bus.record_vector_query(vq2)

    # Record responses for both

    llm_req1 = LLMRequestEvent(
        model="test",
        vector_query_id=vq1.id,
        session_id=session1.id,
    )
    bus.record_llm_request(llm_req1)

    resp1 = LLMResponseEvent(
        output_text="Paris is in France.",
        request_id=llm_req1.id,
        session_id=session1.id,
    )
    bus.record_llm_response(resp1)

    llm_req2 = LLMRequestEvent(
        model="test",
        vector_query_id=vq2.id,
        session_id=session2.id,
    )
    bus.record_llm_request(llm_req2)

    resp2 = LLMResponseEvent(
        output_text="London is in England.",
        request_id=llm_req2.id,
        session_id=session2.id,
    )
    bus.record_llm_response(resp2)

    _run_attribution(resp1, _bus=bus)
    _run_attribution(resp2, _bus=bus)

    # Verify sessions are isolated
    retrieved_session1 = bus.get_session(session1.id)
    retrieved_session2 = bus.get_session(session2.id)

    assert len(retrieved_session1.vector_queries) == 1
    assert len(retrieved_session2.vector_queries) == 1

    # Each session should have its own attribution
    assert len(retrieved_session1.attributions) == 1
    assert len(retrieved_session2.attributions) == 1

    # Attributions should reference different chunks
    attr1_chunk_ids = {c.chunk_id for c in retrieved_session1.attributions[0].chunks}
    attr2_chunk_ids = {c.chunk_id for c in retrieved_session2.attributions[0].chunks}

    assert "c1" in attr1_chunk_ids
    assert "c2" in attr2_chunk_ids
    assert attr1_chunk_ids != attr2_chunk_ids
