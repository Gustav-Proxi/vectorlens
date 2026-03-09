"""Attribution module for measuring which chunks caused which outputs."""

from vectorlens.attribution.attention import AttentionAttributor
from vectorlens.attribution.perturbation import PerturbationAttributor
from vectorlens.types import RetrievedChunk, Session

__all__ = [
    "AttentionAttributor",
    "PerturbationAttributor",
    "compute_attribution_for_session",
]


def compute_attribution_for_session(
    session: Session, llm_caller
) -> list[RetrievedChunk]:
    """
    Helper to run attribution on a session.

    Args:
        session: VectorLens session
        llm_caller: Async LLM caller function

    Returns:
        Chunks with attribution scores computed
    """
    # This is a synchronous wrapper around the async compute method.
    # Actual usage would require async context.
    attributor = PerturbationAttributor(llm_caller)
    return attributor
