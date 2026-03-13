"""GraphRAG-specific attribution for VectorLens.

Solves the "no discrete chunk" problem for GraphRAG global search by treating
community reports as attribution units and using semantic similarity to attribute
hallucinated sentences back to the community reports that most influenced them.

Two-tier design:
  Tier 1 — Semantic Similarity (zero LLM calls, used always):
    For each hallucinated sentence H and community report C:
      score(C, H) = cosine_similarity(embed(H), embed(C.text))
    Attribution = max pooling over hallucinated sentences.
    Interpretation: the community that is semantically closest to what the LLM
    hallucinated most likely provided the "inspiration" that got distorted.

  Tier 2 — Reduce-Stage Perturbation (future, requires captured reduce prompt):
    Remove one community's contribution from the reduce LLM prompt, re-run,
    measure output change. High change = high attribution.
    Not implemented in v1 — Tier 1 already solves the discrete-unit problem.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from vectorlens.types import (
    GraphRAGCommunityUnit,
    GraphRAGContextEvent,
    OutputToken,
    RetrievedChunk,
)

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load sentence-transformers model (shared singleton with perturbation.py)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CommunityAttributor:
    """Attributes hallucinated output back to GraphRAG community reports.

    This solves the attribution problem for GlobalSearch, where retrieved
    content is not discrete text chunks but synthesized community summaries.
    """

    def attribute(
        self,
        output_tokens: list[OutputToken],
        community_units: list[GraphRAGCommunityUnit],
    ) -> list[GraphRAGCommunityUnit]:
        """Score each community unit against hallucinated output sentences.

        Algorithm (Tier 1 — Semantic Similarity):
          1. Extract hallucinated sentences from output_tokens
          2. Embed each hallucinated sentence + each community report
          3. For each community C: score(C) = max sim(C, H) over all hallucinated H
          4. Normalize scores to [0, 1]

        Communities with high scores were semantically close to hallucinated
        content — they provided context the LLM distorted.

        Communities with near-zero scores across ALL hallucinated sentences
        indicate the hallucination was fully fabricated (no community inspired it).

        Args:
            output_tokens: OutputToken list from hallucination detection.
            community_units: Community reports to score.

        Returns:
            community_units with attribution_score set (in-place + returned).
        """
        if not community_units:
            return community_units

        hallucinated_sentences = _extract_hallucinated_sentences(output_tokens)
        if not hallucinated_sentences:
            # No hallucinations — all communities score 0
            return community_units

        model = _get_model()

        # Embed hallucinated sentences
        hall_embs = model.encode(hallucinated_sentences, convert_to_numpy=True)
        # Shape: (n_hallucinated, embed_dim)

        # Embed community report texts
        comm_texts = [u.text for u in community_units]
        if not any(comm_texts):
            return community_units
        comm_embs = model.encode(comm_texts, convert_to_numpy=True)
        # Shape: (n_communities, embed_dim)

        # For each community: max cosine similarity over all hallucinated sentences
        scores = np.zeros(len(community_units))
        for i, c_emb in enumerate(comm_embs):
            for h_emb in hall_embs:
                scores[i] = max(scores[i], _cosine_sim(c_emb, h_emb))

        # Normalize to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        for unit, score in zip(community_units, scores):
            unit.attribution_score = float(score)
            # Flag as causally involved if score is above a meaningful threshold.
            # 0.5 (after normalization) means this community was at least half as
            # similar to the hallucination as the most similar community.
            unit.caused_hallucination = float(score) > 0.5

        logger.debug(
            f"Community attribution: {len(hallucinated_sentences)} hallucinated sentences, "
            f"{len(community_units)} communities, "
            f"top_score={max_score:.3f}, "
            f"flagged={sum(u.caused_hallucination for u in community_units)}"
        )

        return community_units

    def overall_groundedness(
        self,
        output_tokens: list[OutputToken],
        community_units: list[GraphRAGCommunityUnit],
    ) -> float:
        """Estimate groundedness of the output against community reports.

        For each output sentence, compute max similarity to ANY community.
        Groundedness = fraction of sentences with max_sim >= 0.4.

        This mirrors HallucinationDetector's threshold logic but operates
        against community reports rather than retrieved chunks.
        """
        if not community_units or not output_tokens:
            return 1.0

        model = _get_model()
        sentences = _extract_all_sentences(output_tokens)
        if not sentences:
            return 1.0

        comm_texts = [u.text for u in community_units if u.text]
        if not comm_texts:
            return 1.0

        sent_embs = model.encode(sentences, convert_to_numpy=True)
        comm_embs = model.encode(comm_texts, convert_to_numpy=True)

        grounded = 0
        for s_emb in sent_embs:
            max_sim = max(_cosine_sim(s_emb, c_emb) for c_emb in comm_embs)
            if max_sim >= 0.4:
                grounded += 1

        return grounded / len(sentences)


def community_units_to_chunks(
    units: list[GraphRAGCommunityUnit],
) -> list[RetrievedChunk]:
    """Convert community units to RetrievedChunk for API serialization.

    Allows the existing AttributionResult.chunks field to carry community
    attribution data without changing the API response schema.
    """
    return [
        RetrievedChunk(
            chunk_id=unit.unit_id,
            text=unit.text,
            score=unit.rank / max(u.rank for u in units) if units else 0.0,
            attribution_score=unit.attribution_score,
            caused_hallucination=unit.caused_hallucination,
            metadata={
                "type": "graphrag_community",
                "community_id": unit.community_id,
                "title": unit.title,
            },
        )
        for unit in units
    ]


# ------------------------------------------------------------------
# Sentence extraction helpers
# ------------------------------------------------------------------

def _extract_hallucinated_sentences(output_tokens: list[OutputToken]) -> list[str]:
    """Collect text of hallucinated tokens, grouped into sentences."""
    hallucinated_text = " ".join(
        t.text for t in output_tokens if t.is_hallucinated and t.text.strip()
    )
    if not hallucinated_text.strip():
        return []
    # Split into sentences — simple period/question/exclamation split
    import re
    sentences = re.split(r"(?<=[.!?])\s+", hallucinated_text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_all_sentences(output_tokens: list[OutputToken]) -> list[str]:
    """Extract all output sentences from token list."""
    import re
    full_text = " ".join(t.text for t in output_tokens if t.text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", full_text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]
