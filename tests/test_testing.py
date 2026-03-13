"""Tests for vectorlens.testing module."""
from __future__ import annotations

import pytest

from vectorlens.testing import assert_grounded, groundedness_score
from vectorlens.types import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for testing."""
    return RetrievedChunk(chunk_id=chunk_id, text=text, score=1.0)


# ---------------------------------------------------------------------------
# groundedness_score()
# ---------------------------------------------------------------------------

def test_groundedness_score_returns_float():
    """groundedness_score() must return a float between 0.0 and 1.0."""
    chunks = [_make_chunk("c1", "The sky is blue.")]
    score = groundedness_score("The sky is blue.", chunks)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_groundedness_score_empty_response():
    """Empty response should return 1.0 (nothing to hallucinate)."""
    chunks = [_make_chunk("c1", "Some context.")]
    score = groundedness_score("", chunks)
    assert score == 1.0


def test_groundedness_score_no_chunks():
    """No chunks should return 0.0 (nothing to ground against)."""
    score = groundedness_score("The sky is blue.", [])
    assert score == 0.0


# ---------------------------------------------------------------------------
# assert_grounded()
# ---------------------------------------------------------------------------

def test_assert_grounded_passes_when_grounded():
    """assert_grounded() must not raise when response is grounded."""
    chunks = [_make_chunk("c1", "The sky is blue and the sun is bright.")]
    # Should not raise
    assert_grounded("The sky is blue.", chunks, min_score=0.5)


def test_assert_grounded_raises_when_hallucinated():
    """assert_grounded() must raise AssertionError when score is below min_score."""
    chunks = [_make_chunk("c1", "Cats are fluffy animals.")]
    with pytest.raises(AssertionError) as exc_info:
        assert_grounded(
            "Quantum physics governs the universe at subatomic scales.",
            chunks,
            min_score=0.99,
        )
    assert "Groundedness score" in str(exc_info.value)
    assert "Hallucinated sentences" in str(exc_info.value)


def test_assert_grounded_default_min_score():
    """assert_grounded() default min_score must be 0.75."""
    import inspect
    from vectorlens.testing import assert_grounded
    sig = inspect.signature(assert_grounded)
    assert sig.parameters["min_score"].default == 0.75


def test_assert_grounded_custom_message():
    """assert_grounded() must include custom message in AssertionError."""
    chunks = [_make_chunk("c1", "Cats are fluffy.")]
    with pytest.raises(AssertionError) as exc_info:
        assert_grounded(
            "Quantum physics governs subatomic scales.",
            chunks,
            min_score=0.99,
            message="Custom error message",
        )
    assert "Custom error message" in str(exc_info.value)


def test_assert_grounded_empty_response():
    """assert_grounded() must not raise for empty response."""
    chunks = [_make_chunk("c1", "Some context.")]
    assert_grounded("", chunks)