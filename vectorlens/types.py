"""Shared types and data models for VectorLens.

All agents must import from here — do not redefine these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import time
import uuid


class EventType(str, Enum):
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    VECTOR_QUERY = "vector_query"
    VECTOR_RESULTS = "vector_results"
    ATTRIBUTION_COMPUTED = "attribution_computed"
    HALLUCINATION_DETECTED = "hallucination_detected"


@dataclass
class VectorQueryEvent:
    """Fired when a vector DB is queried."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    db_type: str = ""           # "chroma", "pinecone", "faiss", "weaviate"
    collection: str = ""
    query_text: str = ""
    query_embedding: list[float] = field(default_factory=list)
    top_k: int = 0
    results: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A single chunk returned from a vector DB query."""
    chunk_id: str = ""
    text: str = ""
    score: float = 0.0          # similarity score from vector DB
    metadata: dict[str, Any] = field(default_factory=dict)
    # Set after attribution
    attribution_score: float = 0.0   # 0-1, how much this chunk caused the output
    caused_hallucination: bool = False


@dataclass
class LLMRequestEvent:
    """Fired when an LLM is called."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    timestamp: float = field(default_factory=time.time)
    provider: str = ""          # "openai", "anthropic", "transformers"
    model: str = ""
    system_prompt: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 1024
    # Linked to vector query
    vector_query_id: str | None = None
    # Multi-turn conversation linking
    parent_request_id: str | None = None   # ID of the LLM call that triggered this one
    chain_step: str = ""                    # e.g., "retrieval", "generation", "reflection"

    def __post_init__(self) -> None:
        # Prevent memory DoS via unbounded string fields from interceptors
        if self.parent_request_id and len(self.parent_request_id) > 1000:
            self.parent_request_id = self.parent_request_id[:1000]
        if len(self.chain_step) > 256:
            self.chain_step = self.chain_step[:256]


@dataclass
class LLMResponseEvent:
    """Fired when an LLM returns a response."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    output_text: str = ""
    output_tokens: list[OutputToken] = field(default_factory=list)
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class OutputToken:
    """A single token in the LLM output with hallucination metadata."""
    text: str = ""
    position: int = 0
    is_hallucinated: bool = False   # Set by detection module
    hallucination_score: float = 0.0  # 0-1
    # chunk_id -> attribution weight (0-1)
    chunk_attributions: dict[str, float] = field(default_factory=dict)


@dataclass
class AttributionResult:
    """Result of running attribution on an LLM response."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    request_id: str = ""
    response_id: str = ""
    timestamp: float = field(default_factory=time.time)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    output_tokens: list[OutputToken] = field(default_factory=list)
    overall_groundedness: float = 0.0   # 0-1, 1 = fully grounded
    hallucinated_spans: list[tuple[int, int]] = field(default_factory=list)  # (start, end) token indices


@dataclass
class Session:
    """A VectorLens tracing session grouping one RAG pipeline execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    vector_queries: list[VectorQueryEvent] = field(default_factory=list)
    llm_requests: list[LLMRequestEvent] = field(default_factory=list)
    llm_responses: list[LLMResponseEvent] = field(default_factory=list)
    attributions: list[AttributionResult] = field(default_factory=list)
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Per-session HF model reference — set by TransformersInterceptor so
    # attention attribution uses the correct model in concurrent servers
    # (avoids the global _intercepted_model bleed-across-sessions bug)
    hf_model: Any | None = field(default=None, repr=False, compare=False)
    hf_tokenizer: Any | None = field(default=None, repr=False, compare=False)
