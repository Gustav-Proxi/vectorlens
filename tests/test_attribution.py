"""Tests for perturbation-based attribution module.

Tests attribution without requiring real LLM calls or embedding models.
"""
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vectorlens.attribution.perturbation import (
    PerturbationAttributor,
    _cosine_similarity,
    _remove_chunk_from_messages,
)
from vectorlens.types import OutputToken, RetrievedChunk


class TestRemoveChunkFromMessages:
    """Tests for chunk removal helper."""

    def test_remove_chunk_from_messages(self):
        """Test removing a chunk from messages."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Context: Important fact. Question: What?"},
        ]
        chunk_text = "Important fact."

        result = _remove_chunk_from_messages(messages, chunk_text)

        assert len(result) == 2
        assert result[0]["content"] == "You are helpful."
        # Chunk removed from second message
        assert "Important fact." not in result[1]["content"]
        assert "Context:" in result[1]["content"]
        assert "Question:" in result[1]["content"]

    def test_remove_chunk_not_found(self):
        """Test removing chunk that doesn't exist."""
        messages = [{"role": "user", "content": "Original text"}]
        chunk_text = "Not in text"

        result = _remove_chunk_from_messages(messages, chunk_text)

        # Should leave message unchanged
        assert result[0]["content"] == "Original text"

    def test_remove_multiple_occurrences(self):
        """Test removing multiple occurrences of chunk."""
        messages = [
            {
                "role": "user",
                "content": "Fact X. Fact X. Fact X.",
            }
        ]

        result = _remove_chunk_from_messages(messages, "Fact X.")

        # All occurrences removed
        assert "Fact X." not in result[0]["content"]

    def test_preserve_non_string_content(self):
        """Test that non-string content is preserved."""
        messages = [
            {"role": "user", "content": "text"},
            {"role": "assistant", "content": {"type": "data"}},
        ]

        result = _remove_chunk_from_messages(messages, "text")

        assert result[0]["content"] == ""  # Removed
        assert result[1]["content"] == {"type": "data"}  # Preserved


class TestPerturbationAttributor:
    """Tests for perturbation-based attribution."""

    def setup_method(self):
        """Create fresh attributor for each test."""
        self.llm_caller = AsyncMock()
        self.attributor = PerturbationAttributor(self.llm_caller)

    @pytest.mark.asyncio
    async def test_high_attribution_when_chunk_removal_changes_output(self):
        """Test high attribution when removing chunk significantly changes output."""
        fake_model = MagicMock()

        # Original output embedding (1D — matches real sentence-transformers output)
        original_output = "Paris is the capital of France and is beautiful."
        original_emb = np.array([1.0, 0.0, 0.0])

        # When chunk1 removed, output is very different
        perturbed_output = "Unknown location is beautiful."
        perturbed_emb = np.array([0.1, 0.9, 0.0])  # Very different

        fake_model.encode.side_effect = [
            original_emb,  # First call: encode(original_output)
            perturbed_emb,  # Second call: encode(perturbed_output)
        ]

        # Mock LLM to return different output when chunk is removed
        self.llm_caller.return_value = perturbed_output

        with patch(
            "vectorlens.attribution.perturbation._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(
                chunk_id="chunk1",
                text="Paris is the capital of France",
                score=0.9,
            )
            messages = [{"role": "user", "content": original_output}]

            result = await self.attributor.compute(
                messages,
                [chunk],
                original_output,
                [],
            )

            # Should have high attribution
            assert result[0].attribution_score > 0.5
            assert result[0].attribution_score <= 1.0

    @pytest.mark.asyncio
    async def test_low_attribution_when_chunk_not_used(self):
        """Test low attribution when chunk doesn't affect output."""
        fake_model = MagicMock()

        original_output = "The sky is blue."
        perturbed_output = "The sky is blue."  # Same output

        original_emb = np.array([[1.0, 0.0, 0.0]])
        perturbed_emb = np.array([[1.0, 0.0, 0.0]])  # Identical

        fake_model.encode.side_effect = [
            original_emb,
            perturbed_emb,
        ]

        self.llm_caller.return_value = perturbed_output

        with patch(
            "vectorlens.attribution.perturbation._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(
                chunk_id="chunk1",
                text="Unused chunk",
                score=0.5,
            )

            result = await self.attributor.compute(
                [{"role": "user", "content": original_output}],
                [chunk],
                original_output,
                [],
            )

            # Low attribution since output didn't change
            assert result[0].attribution_score < 0.2

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        """Test that LLM failure for one chunk doesn't affect others."""
        fake_model = MagicMock()

        original_output = "Text"
        original_emb = np.array([[1.0, 0.0]])
        working_emb = np.array([[0.9, 0.1]])
        failing_emb = np.array([[0.5, 0.5]])

        fake_model.encode.side_effect = [
            original_emb,  # Original
            working_emb,   # Chunk 1 works
            failing_emb,   # Chunk 2 attempt (will fail)
        ]

        # Mock: first call succeeds, second raises
        self.llm_caller.side_effect = [
            "Modified output 1",
            Exception("LLM timeout"),
        ]

        with patch(
            "vectorlens.attribution.perturbation._get_model", return_value=fake_model
        ):
            chunks = [
                RetrievedChunk(chunk_id="c1", text="chunk1", score=0.9),
                RetrievedChunk(chunk_id="c2", text="chunk2", score=0.8),
            ]

            result = await self.attributor.compute(
                [{"role": "user", "content": "text"}],
                chunks,
                original_output,
                [],
            )

            # First chunk should have attribution
            assert result[0].attribution_score >= 0.0
            # Second chunk failed, gets 0
            assert result[1].attribution_score == 0.0

    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self):
        """Test that empty chunks returns unchanged chunks."""
        result = await self.attributor.compute(
            [],
            [],
            "output",
            [],
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_output_returns_chunks(self):
        """Test that empty output returns chunks with no attribution."""
        chunks = [RetrievedChunk(chunk_id="c1", text="text", score=0.9)]
        result = await self.attributor.compute(
            [{"role": "user", "content": "text"}],
            chunks,
            "",  # Empty output
            [],
        )

        # Should return same chunks, likely with 0 attribution
        assert len(result) == 1
        assert result[0].chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_multiple_chunks_concurrent_perturbation(self):
        """Test that multiple chunks are perturbed concurrently."""
        fake_model = MagicMock()

        original_emb = np.array([[1.0, 0.0, 0.0]])
        emb1 = np.array([[0.9, 0.1, 0.0]])
        emb2 = np.array([[0.8, 0.2, 0.0]])
        emb3 = np.array([[0.7, 0.3, 0.0]])

        fake_model.encode.side_effect = [
            original_emb,
            emb1,
            emb2,
            emb3,
        ]

        outputs = ["out1", "out2", "out3"]
        call_count = 0

        async def mock_llm(messages):
            nonlocal call_count
            result = outputs[call_count]
            call_count += 1
            return result

        self.attributor.llm_caller = mock_llm

        with patch(
            "vectorlens.attribution.perturbation._get_model", return_value=fake_model
        ):
            chunks = [
                RetrievedChunk(chunk_id=f"c{i}", text=f"chunk{i}", score=0.9)
                for i in range(3)
            ]

            result = await self.attributor.compute(
                [{"role": "user", "content": "text"}],
                chunks,
                "original",
                [],
            )

            # All chunks should be processed
            assert len(result) == 3
            # All should have attribution scores
            for chunk in result:
                assert chunk.attribution_score >= 0.0

    @pytest.mark.asyncio
    async def test_perturbed_output_empty_returns_high_attribution(self):
        """Test that completely empty perturbed output gives high attribution."""
        fake_model = MagicMock()

        original_emb = np.array([[1.0, 0.0]])
        # Empty output will skip similarity check and return 1.0

        fake_model.encode.return_value = original_emb

        self.llm_caller.return_value = ""  # Empty output

        with patch(
            "vectorlens.attribution.perturbation._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(chunk_id="c1", text="important", score=0.9)

            result = await self.attributor.compute(
                [{"role": "user", "content": "with important chunk"}],
                [chunk],
                "original output",
                [],
            )

            # Empty output = chunk is essential = high attribution
            assert result[0].attribution_score == 1.0

    @pytest.mark.asyncio
    async def test_clamping_attribution_to_01_range(self):
        """Test that attribution scores are clamped to [0, 1]."""
        fake_model = MagicMock()

        # Create embeddings that would give > 1.0 or < 0.0 if not clamped
        original_emb = np.array([[1.0, 0.0]])
        perturbed_emb = np.array([[-2.0, 0.0]])  # Would give similarity -2.0

        fake_model.encode.side_effect = [
            original_emb,
            perturbed_emb,
        ]

        self.llm_caller.return_value = "perturbed"

        with patch(
            "vectorlens.attribution.perturbation._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(chunk_id="c1", text="text", score=0.9)

            result = await self.attributor.compute(
                [{"role": "user", "content": "text"}],
                [chunk],
                "original",
                [],
            )

            # Should be clamped to [0, 1]
            assert 0.0 <= result[0].attribution_score <= 1.0


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestCosineSimilarity:
    """Tests for cosine similarity in attribution module."""

    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = _cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = _cosine_similarity(a, b)
        assert sim == pytest.approx(0.0)

    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        sim = _cosine_similarity(a, b)
        assert sim == pytest.approx(-1.0)

    def test_normalized_vectors(self):
        """Test similarity with normalized vectors."""
        a = np.array([3.0, 4.0])
        b = np.array([3.0, 4.0])
        sim = _cosine_similarity(a, b)
        assert sim == pytest.approx(1.0)

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = _cosine_similarity(a, b)
        assert sim == 0.0

    def test_high_dimensional_vectors(self):
        """Test similarity with high-dimensional vectors."""
        # Random vectors in 100D
        a = np.random.randn(100)
        b = np.random.randn(100)
        sim = _cosine_similarity(a, b)
        # Should be in [-1, 1]
        assert -1.0 <= sim <= 1.0


class TestMessageRemoval:
    """Additional tests for message chunk removal."""

    def test_removal_with_extra_whitespace(self):
        """Test removal with extra whitespace."""
        messages = [
            {"role": "user", "content": "Before  Chunk  After"}
        ]
        result = _remove_chunk_from_messages(messages, "Chunk")
        assert "Before" in result[0]["content"]
        assert "After" in result[0]["content"]
        assert "Chunk" not in result[0]["content"]

    def test_removal_case_sensitive(self):
        """Test that removal is case-sensitive."""
        messages = [
            {"role": "user", "content": "Important chunk text"}
        ]
        result = _remove_chunk_from_messages(messages, "IMPORTANT")
        # Should not remove (case-sensitive)
        assert "Important" in result[0]["content"]

    def test_multiple_messages_independence(self):
        """Test that removal in one message doesn't affect others."""
        messages = [
            {"role": "system", "content": "Chunk in system"},
            {"role": "user", "content": "Chunk in user"},
            {"role": "assistant", "content": "Chunk in assistant"},
        ]
        result = _remove_chunk_from_messages(messages, "Chunk")

        # All should have "Chunk" removed
        assert "Chunk" not in result[0]["content"]
        assert "Chunk" not in result[1]["content"]
        assert "Chunk" not in result[2]["content"]

    def test_complex_message_structure(self):
        """Test with messages that have additional fields."""
        messages = [
            {
                "role": "user",
                "content": "Text with chunk",
                "metadata": {"timestamp": 123},
            }
        ]
        result = _remove_chunk_from_messages(messages, "chunk")

        # Metadata preserved
        assert "metadata" in result[0]
        assert result[0]["metadata"]["timestamp"] == 123
