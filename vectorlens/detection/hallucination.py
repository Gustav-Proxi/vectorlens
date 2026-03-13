"""Hallucination detection module for VectorLens.

Detects hallucinated content in LLM output by comparing sentence embeddings
against retrieved chunks.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from vectorlens.types import OutputToken, RetrievedChunk


# Lazy-loaded model singleton
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load sentence-transformers model as singleton."""
    global _model
    if _model is None:
        import sys
        from pathlib import Path

        # Check if model is already cached
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_cached = any(
            (cache_dir / d).exists()
            for d in [model_name.replace("/", "--"), f"models--{model_name.replace('/', '--')}"]
        )

        if not model_cached:
            print(
                "\033[33m[VectorLens] Downloading all-MiniLM-L6-v2 (~90MB) for "
                "hallucination detection — one-time download...\033[0m",
                file=sys.stderr,
            )

        _model = SentenceTransformer("all-MiniLM-L6-v2")

        if not model_cached:
            print("\033[32m[VectorLens] Model ready.\033[0m", file=sys.stderr)

    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """
    Split text into sentences.

    Returns:
        List of (sentence, start_pos, end_pos) tuples.
    """
    if not text:
        return []

    # Simple sentence splitting on ". " and final period
    sentences = []
    current_pos = 0

    # Split on ". " first
    parts = text.split(". ")
    for i, part in enumerate(parts):
        if not part.strip():
            current_pos += len(part) + 2  # Account for ". "
            continue

        # Add back the period except for the last part if it doesn't have one
        if i < len(parts) - 1:
            sentence = part + "."
        else:
            # Last part: check if it ends with period
            sentence = part if part.endswith(".") else part

        start_pos = current_pos
        end_pos = start_pos + len(sentence)
        sentences.append((sentence.strip(), start_pos, end_pos))
        current_pos = end_pos + 2  # Account for ". "

    return sentences


class HallucinationDetector:
    """Detects hallucinated tokens in LLM output."""

    HALLUCINATION_THRESHOLD: float = 0.4

    def __init__(self) -> None:
        """Initialize the detector. Model is lazy-loaded on first use."""
        pass

    def detect(
        self,
        output_text: str,
        chunks: list[RetrievedChunk],
    ) -> list[OutputToken]:
        """
        Detect hallucinations in output text.

        Splits output into sentences, computes embeddings, and compares against
        chunk embeddings to detect hallucinated content.

        Args:
            output_text: The LLM output text to analyze
            chunks: List of retrieved chunks from the vector store

        Returns:
            List of OutputToken objects with hallucination detection results
        """
        # Handle edge cases
        if not output_text or not output_text.strip():
            return []

        if not chunks:
            # No chunks to compare against — treat all sentences as hallucinated
            sentences = _split_sentences(output_text)
            return [
                OutputToken(
                    text=sent,
                    position=i,
                    is_hallucinated=True,
                    hallucination_score=1.0,
                    chunk_attributions={},
                )
                for i, (sent, _, _) in enumerate(sentences)
            ]

        # Get model
        model = _get_model()

        # Split into sentences
        sentences = _split_sentences(output_text)
        if not sentences:
            return []

        # Compute embeddings
        sentence_texts = [sent for sent, _, _ in sentences]
        sentence_embeddings = model.encode(sentence_texts, convert_to_numpy=True)

        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = model.encode(chunk_texts, convert_to_numpy=True)

        # Detect hallucinations for each sentence
        output_tokens: list[OutputToken] = []

        for position, (sentence, start_pos, end_pos) in enumerate(sentences):
            sent_embedding = sentence_embeddings[position]

            # Find max similarity to any chunk
            max_similarity = 0.0
            top_chunks: dict[str, float] = {}

            # Compute similarity to all chunks
            similarities: list[tuple[str, float]] = []
            for chunk, chunk_emb in zip(chunks, chunk_embeddings):
                sim = _cosine_similarity(sent_embedding, chunk_emb)
                similarities.append((chunk.chunk_id, sim))
                max_similarity = max(max_similarity, sim)

            # Determine if hallucinated
            is_hallucinated = max_similarity < self.HALLUCINATION_THRESHOLD
            hallucination_score = 1.0 - max_similarity if is_hallucinated else 0.0

            # Get top-3 most similar chunks
            if not is_hallucinated:
                similarities.sort(key=lambda x: x[1], reverse=True)
                for chunk_id, sim in similarities[:3]:
                    top_chunks[chunk_id] = sim

            token = OutputToken(
                text=sentence,
                position=position,
                is_hallucinated=is_hallucinated,
                hallucination_score=hallucination_score,
                chunk_attributions=top_chunks,
            )
            output_tokens.append(token)

        return output_tokens

