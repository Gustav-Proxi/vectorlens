"""Testing utilities for VectorLens.

Provides inline RAG quality assertions for use in pytest, Jupyter notebooks,
or plain Python scripts — no server required.
"""
from __future__ import annotations

from typing import Optional

from vectorlens.detection.hallucination import HallucinationDetector
from vectorlens.types import RetrievedChunk


def groundedness_score(
    response_text: str,
    chunks: list[RetrievedChunk],
) -> float:
    """
    Compute the groundedness score of a response against retrieved chunks.

    Parameters
    ----------
    response_text : str
        The LLM response text to evaluate.

    chunks : list[RetrievedChunk]
        The retrieved chunks to compare against.

    Returns
    -------
    float
        A score between 0.0 and 1.0, where 1.0 means fully grounded.
    """
    if not response_text or not response_text.strip():
        return 1.0

    detector = HallucinationDetector()
    tokens = detector.detect(response_text, chunks)

    if not tokens:
        return 1.0

    grounded = sum(1 for t in tokens if not t.is_hallucinated)
    return grounded / len(tokens)


def assert_grounded(
    response_text: str,
    chunks: list[RetrievedChunk],
    min_score: float = 0.75,
    message: Optional[str] = None,
) -> None:
    """
    Assert that a RAG response is grounded in the retrieved chunks.

    Parameters
    ----------
    response_text : str
        The LLM response text to evaluate.

    chunks : list[RetrievedChunk]
        The retrieved chunks to compare against.

    min_score : float, optional, default=0.75
        Minimum groundedness score (0.0 to 1.0) required to pass.

    message : str, optional
        Custom message to include in the AssertionError.

    Raises
    ------
    AssertionError
        If the groundedness score is below min_score, with a diff
        showing which sentences were hallucinated.
    """
    detector = HallucinationDetector()
    tokens = detector.detect(response_text, chunks)

    if not tokens:
        return

    hallucinated = [t for t in tokens if t.is_hallucinated]
    grounded_count = len(tokens) - len(hallucinated)
    score = grounded_count / len(tokens)

    if score < min_score:
        diff_lines = [
            f"Groundedness score {score:.2f} is below minimum {min_score:.2f}",
            f"  Grounded sentences : {grounded_count}/{len(tokens)}",
            f"  Hallucinated sentences: {len(hallucinated)}/{len(tokens)}",
            "",
            "Hallucinated sentences:",
        ]
        for t in hallucinated:
            diff_lines.append(f"  - [{t.position}] {t.text!r}")

        diff = "\n".join(diff_lines)
        if message:
            diff = f"{message}\n{diff}"

        raise AssertionError(diff)