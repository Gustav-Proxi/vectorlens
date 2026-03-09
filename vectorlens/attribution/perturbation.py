"""Perturbation-based attribution for VectorLens.

Measures which retrieved chunks caused which parts of the output by dropping
each chunk and measuring output divergence. Also supports LIME-style bounded
perturbation for fixed-cost attribution.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Callable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from vectorlens.types import OutputToken, RetrievedChunk

logger = logging.getLogger(__name__)


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
                "attribution scoring — one-time download...\033[0m",
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


def _remove_chunk_from_messages(
    messages: list[dict], chunk_text: str
) -> list[dict]:
    """Remove a chunk from messages using robust semantic matching.

    Tries in order:
    1. Exact substring match (fast path)
    2. First-N-chars prefix match (handles truncated storage)
    3. First-sentence match (handles metadata wrapping)

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        chunk_text: Text of the chunk to remove

    Returns:
        New messages list with chunk removed (or unchanged if no match found)
    """
    new_messages = []
    chunk_prefix = chunk_text[:120].strip()  # first ~120 chars for truncation
    chunk_first_sentence = chunk_text.split(".")[0][:80].strip()  # first sentence

    for msg in messages:
        new_msg = msg.copy()
        content = new_msg.get("content")
        if not isinstance(content, str):
            new_messages.append(new_msg)
            continue

        # 1. Exact match (fast path)
        if chunk_text in content:
            new_msg["content"] = content.replace(chunk_text, "").strip()
        # 2. Prefix match (handles truncated chunks)
        elif len(chunk_prefix) > 30 and chunk_prefix in content:
            idx = content.find(chunk_prefix)
            # Remove from idx to next double-newline or end
            end = content.find("\n\n", idx)
            if end == -1:
                end = len(content)
            new_msg["content"] = (content[:idx] + content[end:]).strip()
        # 3. First-sentence match (handles metadata wrapping)
        elif len(chunk_first_sentence) > 20 and chunk_first_sentence in content:
            idx = content.find(chunk_first_sentence)
            end = content.find("\n\n", idx)
            if end == -1:
                end = len(content)
            new_msg["content"] = (content[:idx] + content[end:]).strip()
        # 4. No match — leave unchanged (safer than corrupting context)

        new_messages.append(new_msg)

    return new_messages


def _remove_all_chunks_from_messages(
    messages: list[dict], chunks: list[RetrievedChunk]
) -> list[dict]:
    """
    Remove all chunk texts from messages using robust matching.

    Args:
        messages: List of message dicts
        chunks: List of chunks to remove

    Returns:
        New messages list with all chunks removed
    """
    result = messages
    for chunk in chunks:
        # Use full chunk text with robust semantic matching
        result = _remove_chunk_from_messages(result, chunk.text)
    return result


def _inject_chunk_text(messages: list[dict], text: str) -> list[dict]:
    """
    Append chunk text to the last user message.

    Args:
        messages: List of message dicts
        text: Text to inject

    Returns:
        New messages list with text appended to last user message
    """
    result = [m.copy() for m in messages]
    for msg in reversed(result):
        if msg.get("role") == "user" and isinstance(msg.get("content"), str):
            msg["content"] = msg["content"] + "\n" + text
            break
    return result


N_LIME_SAMPLES = 7  # Fixed cost regardless of chunk count


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

    async def compute_lime(
        self,
        original_messages: list[dict],
        chunks: list[RetrievedChunk],
        original_output: str,
        n_samples: int = N_LIME_SAMPLES,
    ) -> list[RetrievedChunk]:
        """
        LIME-style attribution: fixed K LLM calls regardless of chunk count.

        Algorithm:
        1. Generate n_samples random binary masks over chunks (1=include, 0=drop)
        2. For each mask: build messages with only masked-in chunks, call LLM
        3. Measure semantic similarity between each masked output and original
        4. Fit ridge regression: mask_vectors → similarity_scores
        5. Regression coefficients ≈ per-chunk importance weights
        6. Normalize to [0, 1] and assign as attribution_score

        Cost: exactly n_samples LLM calls, not len(chunks) calls.

        Args:
            original_messages: Original messages passed to LLM
            chunks: List of retrieved chunks
            original_output: Original LLM output text
            n_samples: Number of LIME samples (default 7)

        Returns:
            Chunks with attribution_score set
        """
        if not chunks or not original_output:
            return chunks

        n = len(chunks)
        if n == 0:
            return chunks

        # Embed original output once
        model = _get_model()
        original_emb = model.encode(original_output, convert_to_numpy=True)

        # Generate random masks (n_samples × n_chunks binary matrix)
        # Ensure each chunk appears in at least some masks
        rng = np.random.default_rng(seed=42)
        masks = rng.integers(0, 2, size=(n_samples, n))  # shape: (K, N)

        # Guarantee no all-zero mask (would drop all context)
        for i in range(n_samples):
            if masks[i].sum() == 0:
                masks[i, rng.integers(0, n)] = 1

        # Run LLM for each mask concurrently
        async def _run_masked(mask: np.ndarray) -> float:
            # Build messages with only masked-in chunks
            included = [chunks[j] for j in range(n) if mask[j] == 1]
            if not included:
                return 0.0
            masked_msgs = _remove_all_chunks_from_messages(original_messages, chunks)
            # Re-inject only the included chunks
            for chunk in included:
                masked_msgs = _inject_chunk_text(masked_msgs, chunk.text)

            try:
                masked_output = await self.llm_caller(masked_msgs)
                if not masked_output:
                    return 0.0
                masked_emb = model.encode(masked_output, convert_to_numpy=True)
                return float(_cosine_similarity(original_emb, masked_emb))
            except Exception:
                return 0.0

        similarities = await asyncio.gather(
            *[_run_masked(masks[i]) for i in range(n_samples)]
        )
        similarities = np.array(similarities)  # shape: (K,)

        # Ridge regression: masks (K×N) → similarities (K,)
        # coefficients (N,) = per-chunk attribution
        X = masks.astype(float)
        # Add small L2 regularization
        XtX = X.T @ X + 0.1 * np.eye(n)
        Xty = X.T @ similarities
        coefs = np.linalg.solve(XtX, Xty)  # shape: (N,)

        # Normalize to [0, 1]
        coefs = np.clip(coefs, 0, None)  # attribution can't be negative
        if coefs.max() > 0:
            coefs = coefs / coefs.max()

        for i, chunk in enumerate(chunks):
            chunk.attribution_score = float(coefs[i])

        logger.debug(
            f"LIME attribution: {n_samples} samples for {n} chunks, "
            f"mean similarity={similarities.mean():.3f}"
        )

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
