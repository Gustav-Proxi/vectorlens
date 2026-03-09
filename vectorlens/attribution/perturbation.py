"""Perturbation-based attribution for VectorLens.

Measures which retrieved chunks caused which parts of the output by dropping
each chunk and measuring output divergence.
"""
from __future__ import annotations

import asyncio
import re
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from vectorlens.types import OutputToken, RetrievedChunk


# Lazy-loaded model singleton
_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    """Lazy-load sentence-transformers model as singleton."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def _remove_chunk_from_messages(
    messages: list[dict], chunk_text: str
) -> list[dict]:
    """
    Remove a chunk from the messages by finding and removing its text.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        chunk_text: Text of the chunk to remove

    Returns:
        New messages list with chunk removed
    """
    new_messages = []

    for msg in messages:
        new_msg = msg.copy()
        if "content" in new_msg and isinstance(new_msg["content"], str):
            # Simple string replacement: remove the chunk text
            new_msg["content"] = new_msg["content"].replace(chunk_text, "").strip()

        new_messages.append(new_msg)

    return new_messages


class PerturbationAttributor:
    """Measures attribution via perturbation: drop each chunk and measure output change."""

    def __init__(self, llm_caller: Callable) -> None:
        """
        Initialize the attributor.

        Args:
            llm_caller: Async function(messages: list[dict]) -> str
                       Should call the LLM and return the output text.
        """
        self.llm_caller = llm_caller

    async def compute(
        self,
        original_messages: list[dict],
        chunks: list[RetrievedChunk],
        original_output: str,
        output_tokens: list[OutputToken],
    ) -> list[RetrievedChunk]:
        """
        Compute attribution scores via perturbation.

        For each chunk:
        1. Remove it from the messages
        2. Call llm_caller to get output without that chunk
        3. Compute semantic similarity between original and perturbed output
        4. attribution_score = 1 - similarity (higher change = higher attribution)

        Args:
            original_messages: Original messages passed to LLM
            chunks: List of retrieved chunks
            original_output: Original LLM output text
            output_tokens: Output tokens (not modified, passed for context)

        Returns:
            Chunks with attribution_score set
        """
        if not chunks or not original_output:
            # Edge case: no chunks or empty output
            return chunks

        # Get model for embeddings
        model = _get_model()

        # Encode original output
        original_embedding = model.encode(original_output, convert_to_numpy=True)

        # Run perturbations concurrently
        perturbation_tasks = [
            self._perturb_chunk(
                chunk, original_messages, original_embedding, model
            )
            for chunk in chunks
        ]

        # Gather results
        results = await asyncio.gather(*perturbation_tasks, return_exceptions=True)

        # Assign attribution scores
        for chunk, result in zip(chunks, results):
            if isinstance(result, Exception):
                # If perturbation failed, skip (set score to 0)
                chunk.attribution_score = 0.0
            else:
                chunk.attribution_score = result

        return chunks

    async def _perturb_chunk(
        self,
        chunk: RetrievedChunk,
        original_messages: list[dict],
        original_embedding: np.ndarray,
        model: SentenceTransformer,
    ) -> float:
        """
        Perturb a single chunk and compute attribution.

        Args:
            chunk: The chunk to perturb (remove)
            original_messages: Original messages
            original_embedding: Embedding of original output
            model: SentenceTransformer model for embeddings

        Returns:
            Attribution score (0-1)
        """
        try:
            # Remove chunk from messages
            perturbed_messages = _remove_chunk_from_messages(
                original_messages, chunk.text
            )

            # Call LLM with perturbed messages
            perturbed_output = await self.llm_caller(perturbed_messages)

            if not perturbed_output:
                # If output is empty, chunk was essential
                return 1.0

            # Compute similarity between original and perturbed
            perturbed_embedding = model.encode(
                perturbed_output, convert_to_numpy=True
            )
            similarity = _cosine_similarity(original_embedding, perturbed_embedding)

            # Attribution = change in output (1 - similarity)
            attribution_score = 1.0 - similarity

            # Clamp to [0, 1]
            return max(0.0, min(1.0, attribution_score))

        except Exception:
            # If LLM call fails, skip this perturbation
            return 0.0
