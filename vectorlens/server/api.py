"""REST API endpoints for VectorLens."""
from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_serializer

from vectorlens.interceptors import get_installed
from vectorlens.session_bus import bus
from vectorlens.types import AttributionResult, Session

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================================
# Response Models (Pydantic v2)
# ============================================================================


class SessionSummary(BaseModel):
    """Summary of a session for list view."""

    id: str
    created_at: float
    vector_queries_count: int = 0
    llm_requests_count: int = 0
    llm_responses_count: int = 0
    attributions_count: int = 0

    @staticmethod
    def from_session(session: Session) -> SessionSummary:
        return SessionSummary(
            id=session.id,
            created_at=session.created_at,
            vector_queries_count=len(session.vector_queries),
            llm_requests_count=len(session.llm_requests),
            llm_responses_count=len(session.llm_responses),
            attributions_count=len(session.attributions),
        )


class RetrievedChunkData(BaseModel):
    """Data model for retrieved chunks."""

    chunk_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    attribution_score: float = 0.0
    caused_hallucination: bool = False


class OutputTokenData(BaseModel):
    """Data model for output tokens."""

    text: str
    position: int
    is_hallucinated: bool = False
    hallucination_score: float = 0.0
    chunk_attributions: dict[str, float] = Field(default_factory=dict)


class VectorQueryEventData(BaseModel):
    """Data model for vector query events."""

    id: str
    session_id: str
    timestamp: float
    db_type: str
    collection: str
    query_text: str
    query_embedding: list[float] = Field(default_factory=list)
    top_k: int
    results: list[RetrievedChunkData] = Field(default_factory=list)

    @staticmethod
    def from_event(event: Any) -> VectorQueryEventData:
        return VectorQueryEventData(
            id=event.id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            db_type=event.db_type,
            collection=event.collection,
            query_text=event.query_text,
            query_embedding=event.query_embedding,
            top_k=event.top_k,
            results=[
                RetrievedChunkData(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=c.score,
                    metadata=c.metadata,
                    attribution_score=c.attribution_score,
                    caused_hallucination=c.caused_hallucination,
                )
                for c in event.results
            ],
        )


class LLMRequestEventData(BaseModel):
    """Data model for LLM request events."""

    id: str
    session_id: str
    timestamp: float
    provider: str
    model: str
    system_prompt: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float
    max_tokens: int
    vector_query_id: str | None = None
    # Conversation DAG fields
    parent_request_id: str | None = None
    chain_step: str = ""

    @staticmethod
    def from_event(event: Any) -> LLMRequestEventData:
        return LLMRequestEventData(
            id=event.id,
            session_id=event.session_id,
            timestamp=event.timestamp,
            provider=event.provider,
            model=event.model,
            system_prompt=event.system_prompt,
            messages=event.messages,
            temperature=event.temperature,
            max_tokens=event.max_tokens,
            vector_query_id=event.vector_query_id,
            parent_request_id=getattr(event, "parent_request_id", None),
            chain_step=getattr(event, "chain_step", ""),
        )


class LLMResponseEventData(BaseModel):
    """Data model for LLM response events."""

    id: str
    session_id: str
    request_id: str
    timestamp: float
    output_text: str
    output_tokens: list[OutputTokenData] = Field(default_factory=list)
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float

    @staticmethod
    def from_event(event: Any) -> LLMResponseEventData:
        return LLMResponseEventData(
            id=event.id,
            session_id=event.session_id,
            request_id=event.request_id,
            timestamp=event.timestamp,
            output_text=event.output_text,
            output_tokens=[
                OutputTokenData(
                    text=t.text,
                    position=t.position,
                    is_hallucinated=t.is_hallucinated,
                    hallucination_score=t.hallucination_score,
                    chunk_attributions=t.chunk_attributions,
                )
                for t in event.output_tokens
            ],
            latency_ms=event.latency_ms,
            prompt_tokens=event.prompt_tokens,
            completion_tokens=event.completion_tokens,
            cost_usd=event.cost_usd,
        )


class AttributionData(BaseModel):
    """Data model for attribution results."""

    id: str
    session_id: str
    request_id: str
    response_id: str
    timestamp: float
    chunks: list[RetrievedChunkData] = Field(default_factory=list)
    output_tokens: list[OutputTokenData] = Field(default_factory=list)
    overall_groundedness: float
    hallucinated_spans: list[tuple[int, int]] = Field(default_factory=list)

    @staticmethod
    def from_result(result: AttributionResult) -> AttributionData:
        return AttributionData(
            id=result.id,
            session_id=result.session_id,
            request_id=result.request_id,
            response_id=result.response_id,
            timestamp=result.timestamp,
            chunks=[
                RetrievedChunkData(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    score=c.score,
                    metadata=c.metadata,
                    attribution_score=c.attribution_score,
                    caused_hallucination=c.caused_hallucination,
                )
                for c in result.chunks
            ],
            output_tokens=[
                OutputTokenData(
                    text=t.text,
                    position=t.position,
                    is_hallucinated=t.is_hallucinated,
                    hallucination_score=t.hallucination_score,
                    chunk_attributions=t.chunk_attributions,
                )
                for t in result.output_tokens
            ],
            overall_groundedness=result.overall_groundedness,
            hallucinated_spans=result.hallucinated_spans,
        )


class SessionDetail(BaseModel):
    """Full session detail with all events."""

    id: str
    created_at: float
    conversation_id: str = ""
    vector_queries: list[VectorQueryEventData] = Field(default_factory=list)
    llm_requests: list[LLMRequestEventData] = Field(default_factory=list)
    llm_responses: list[LLMResponseEventData] = Field(default_factory=list)
    attributions: list[AttributionData] = Field(default_factory=list)

    @staticmethod
    def from_session(session: Session) -> SessionDetail:
        return SessionDetail(
            id=session.id,
            created_at=session.created_at,
            conversation_id=getattr(session, "conversation_id", ""),
            vector_queries=[VectorQueryEventData.from_event(q) for q in session.vector_queries],
            llm_requests=[LLMRequestEventData.from_event(r) for r in session.llm_requests],
            llm_responses=[LLMResponseEventData.from_event(r) for r in session.llm_responses],
            attributions=[AttributionData.from_result(a) for a in session.attributions],
        )


class StatusResponse(BaseModel):
    """Server status response."""

    status: str = "ok"
    interceptors: list[str] = Field(default_factory=list)
    active_session: str | None = None


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get server status, installed interceptors, and active session."""
    # active_session is now per-context (ContextVar) — report most recently
    # created session globally so the dashboard always shows the latest run
    active_session = bus._session_order[-1] if bus._session_order else None
    installed = get_installed()
    return StatusResponse(
        status="ok",
        interceptors=installed,
        active_session=active_session,
    )


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions() -> list[SessionSummary]:
    """List all sessions."""
    sessions = bus.all_sessions()
    return [SessionSummary.from_session(s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: str) -> SessionDetail:
    """Get full details of a session."""
    session = bus.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return SessionDetail.from_session(session)


@router.get("/sessions/{session_id}/attributions", response_model=list[AttributionData])
async def get_session_attributions(session_id: str) -> list[AttributionData]:
    """Get attributions for a session."""
    session = bus.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    return [AttributionData.from_result(a) for a in session.attributions]


@router.post("/sessions/new", response_model=SessionSummary)
async def create_session() -> SessionSummary:
    """Create a new session and set it as active."""
    session = bus.new_session()
    return SessionSummary.from_session(session)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str) -> None:
    """Delete a session."""
    deleted = bus.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
