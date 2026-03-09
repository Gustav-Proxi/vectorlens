"""Tests for hallucination detection module.

Tests hallucination detection without requiring actual sentence-transformers.
"""
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vectorlens.detection.hallucination import (
    HallucinationDetector,
    _cosine_similarity,
    _split_sentences,
)
from vectorlens.types import OutputToken, RetrievedChunk


class TestHallucinationDetector:
    """Tests for HallucinationDetector."""

    def setup_method(self):
        """Create fresh detector for each test."""
        self.detector = HallucinationDetector()

    def test_empty_output_returns_empty(self):
        """Test that empty output returns empty list."""
        result = self.detector.detect("", [])
        assert result == []

        result = self.detector.detect("   ", [])
        assert result == []

    def test_empty_chunks_marks_all_hallucinated(self):
        """Test that with no chunks, all output is marked hallucinated."""
        output = "Paris is the capital of France. The sky is blue."
        result = self.detector.detect(output, [])

        # Should have results for each sentence
        assert len(result) > 0

        # All should be marked hallucinated
        for token in result:
            assert token.is_hallucinated is True
            assert token.hallucination_score > 0
            assert token.chunk_attributions == {}

    def test_grounded_sentence_not_hallucinated(self):
        """Test that similar sentence is not marked hallucinated."""
        # Mock the embedding model to return similar embeddings for matching sentences
        fake_model = MagicMock()

        # Setup: output sentence embedding matches chunk embedding
        output_embedding = np.array([[1.0, 0.0, 0.0]])
        chunk_embedding = np.array([[1.0, 0.0, 0.0]])  # Identical = cosine sim 1.0
        fake_model.encode.side_effect = [
            output_embedding,  # First call for output
            chunk_embedding,  # Second call for chunks
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(
                chunk_id="chunk1",
                text="Paris is the capital of France.",
                score=0.95,
            )
            result = self.detector.detect(
                "Paris is the capital of France.",
                [chunk],
            )

            assert len(result) == 1
            token = result[0]
            assert token.is_hallucinated is False
            assert token.hallucination_score == 0.0
            # Should have attribution
            assert len(token.chunk_attributions) > 0

    def test_hallucinated_sentence_detected(self):
        """Test that dissimilar sentence is marked hallucinated."""
        fake_model = MagicMock()

        # Setup: output and chunk embeddings are orthogonal (cosine sim = 0)
        output_embedding = np.array([[1.0, 0.0, 0.0]])
        chunk_embedding = np.array([[0.0, 1.0, 0.0]])  # Orthogonal = cosine sim 0.0
        fake_model.encode.side_effect = [
            output_embedding,
            chunk_embedding,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(
                chunk_id="chunk1",
                text="Unrelated chunk",
                score=0.5,
            )
            result = self.detector.detect(
                "Elephants can fly.",
                [chunk],
            )

            assert len(result) == 1
            token = result[0]
            assert token.is_hallucinated is True
            assert token.hallucination_score > 0
            assert token.chunk_attributions == {}  # No good matches

    def test_chunk_attributions_set_for_grounded(self):
        """Test that chunk attributions are populated for grounded sentences."""
        fake_model = MagicMock()

        # Output similar to chunk 1 and 2, less to chunk 3
        output_embedding = np.array([[1.0, 0.5, 0.0]])
        chunk1_embedding = np.array([[1.0, 0.5, 0.0]])  # sim ≈ 1.0
        chunk2_embedding = np.array([[1.0, 0.4, 0.0]])  # sim ≈ 0.99
        chunk3_embedding = np.array([[0.1, 0.1, 1.0]])  # sim ≈ 0.0

        fake_model.encode.side_effect = [
            output_embedding,  # First call for output
            np.array([chunk1_embedding[0], chunk2_embedding[0], chunk3_embedding[0]]),  # All chunks at once
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunks = [
                RetrievedChunk(chunk_id="chunk1", text="text1", score=0.9),
                RetrievedChunk(chunk_id="chunk2", text="text2", score=0.8),
                RetrievedChunk(chunk_id="chunk3", text="text3", score=0.5),
            ]

            result = self.detector.detect("Test output", chunks)

            token = result[0]
            # Should have top-3 attributions
            assert len(token.chunk_attributions) <= 3
            assert all(isinstance(v, float) for v in token.chunk_attributions.values())

    def test_multiple_sentences(self):
        """Test detection across multiple sentences."""
        fake_model = MagicMock()

        # Two sentences: one grounded, one hallucinated
        sent1_emb = np.array([[1.0, 0.0]])  # Will match chunk
        sent2_emb = np.array([[0.0, 1.0]])  # Won't match chunk
        chunk_emb = np.array([[1.0, 0.0]])

        fake_model.encode.side_effect = [
            np.array([sent1_emb[0], sent2_emb[0]]),  # Output sentences
            np.array([chunk_emb[0]]),  # Chunks
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(chunk_id="c1", text="real fact", score=0.9)
            result = self.detector.detect(
                "Paris is real. Elephants fly.",
                [chunk],
            )

            assert len(result) == 2
            assert result[0].is_hallucinated is False  # Grounded
            assert result[1].is_hallucinated is True  # Hallucinated

    def test_threshold_boundary(self):
        """Test hallucination threshold boundary conditions."""
        fake_model = MagicMock()

        threshold = self.detector.HALLUCINATION_THRESHOLD

        # Test exactly at threshold: similarity = threshold → not hallucinated
        output_emb = np.array([[1.0, 0.0]])
        # Create embedding that will give exactly threshold similarity
        chunk_emb = np.array([[threshold, np.sqrt(1 - threshold**2)]])

        fake_model.encode.side_effect = [
            output_emb,
            chunk_emb,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(chunk_id="c1", text="test", score=0.8)
            result = self.detector.detect("test", [chunk])

            # At threshold → not hallucinated
            assert result[0].is_hallucinated is False

    def test_sentence_splitting(self):
        """Test sentence splitting functionality."""
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)

        assert len(sentences) == 3
        assert sentences[0][0] == "First sentence."
        assert sentences[1][0] == "Second sentence."
        assert sentences[2][0] == "Third sentence."

    def test_sentence_splitting_with_newlines(self):
        """Test sentence splitting with various punctuation."""
        text = "One. Two? Three! Four."
        sentences = _split_sentences(text)

        # Should split on ". "
        assert len(sentences) > 0

    def test_empty_sentences_after_split(self):
        """Test handling of empty sentences."""
        text = "One. . Two."
        sentences = _split_sentences(text)

        # Should skip empty parts
        assert all(s[0].strip() for s in sentences)

    def test_positional_information_preserved(self):
        """Test that sentence positions are correctly recorded."""
        text = "First. Second."
        sentences = _split_sentences(text)

        assert len(sentences) == 2
        first_sent, first_start, first_end = sentences[0]
        assert first_start == 0
        assert first_end >= len("First.")

    def test_multiple_chunks_attribution(self):
        """Test attribution scores with multiple chunks."""
        fake_model = MagicMock()

        output_emb = np.array([[1.0, 0.0, 0.0]])
        chunk1_emb = np.array([[1.0, 0.0, 0.0]])  # sim = 1.0
        chunk2_emb = np.array([[0.99, 0.1, 0.0]])  # sim ≈ 0.98
        chunk3_emb = np.array([[0.8, 0.5, 0.0]])  # sim ≈ 0.93

        fake_model.encode.side_effect = [
            output_emb,
            np.array([chunk1_emb[0], chunk2_emb[0], chunk3_emb[0]]),
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunks = [
                RetrievedChunk(chunk_id="c1", text="text1", score=0.9),
                RetrievedChunk(chunk_id="c2", text="text2", score=0.85),
                RetrievedChunk(chunk_id="c3", text="text3", score=0.8),
            ]

            result = self.detector.detect("test", chunks)
            token = result[0]

            # Should have attributions, sorted by similarity
            assert len(token.chunk_attributions) <= 3
            # Should be in descending order of similarity
            sim_values = list(token.chunk_attributions.values())
            assert sim_values == sorted(sim_values, reverse=True)


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestCosineSimplarity:
    """Tests for cosine similarity calculation."""

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

    def test_zero_vector(self):
        """Test similarity with zero vector."""
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = _cosine_similarity(a, b)
        assert sim == 0.0

    def test_normalized_vectors(self):
        """Test similarity with normalized vectors."""
        a = np.array([3.0, 4.0])  # Length 5
        b = np.array([3.0, 4.0])  # Length 5
        sim = _cosine_similarity(a, b)
        # (3*3 + 4*4) / (5 * 5) = 25 / 25 = 1.0
        assert sim == pytest.approx(1.0)

    def test_partial_similarity(self):
        """Test partial similarity."""
        a = np.array([1.0, 0.0])
        b = np.array([0.707, 0.707])  # 45-degree angle
        sim = _cosine_similarity(a, b)
        assert sim == pytest.approx(0.707, abs=0.01)


class TestSentenceSplitting:
    """Tests for sentence splitting."""

    def test_basic_splitting(self):
        """Test basic sentence splitting."""
        text = "Hello world. How are you."
        sentences = _split_sentences(text)
        assert len(sentences) == 2

    def test_single_sentence_no_period(self):
        """Test single sentence without period."""
        text = "Hello world"
        sentences = _split_sentences(text)
        assert len(sentences) == 1
        assert sentences[0][0] == "Hello world"

    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        text = "First.  Second."
        sentences = _split_sentences(text)
        # Should still split correctly
        assert len(sentences) >= 1

    def test_preserve_content(self):
        """Test that all content is preserved."""
        text = "One. Two. Three."
        sentences = _split_sentences(text)
        combined = " ".join(s[0] for s in sentences)
        # Should contain all original words
        assert "One" in combined
        assert "Two" in combined
        assert "Three" in combined
