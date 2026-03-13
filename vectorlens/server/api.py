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


class TokenHeatmapEntryData(BaseModel):
    """Data model for a single output subword token in the attention heatmap."""

    text: str
    position: int
    chunk_attributions: dict[str, float] = Field(default_factory=dict)


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
    token_heatmap: list[TokenHeatmapEntryData] = Field(default_factory=list)

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
            token_heatmap=[
                TokenHeatmapEntryData(
                    text=e.text,
                    position=e.position,
                    chunk_attributions=e.chunk_attributions,
                )
                for e in getattr(result, "token_heatmap", [])
            ],
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


@router.post("/sessions/{session_id}/analyze", response_model=AttributionData)
async def deep_analyze_session(session_id: str, request_id: str | None = None) -> AttributionData:
    """Run LIME perturbation deep attribution on demand for API models.

    Makes N_LIME_SAMPLES (7) extra LLM calls to measure per-chunk causal influence.
    Updates the stored attribution and returns the result.
    """
    session = bus.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")

    # Find the target LLM request
    llm_request = None
    if request_id:
        llm_request = next((r for r in session.llm_requests if r.id == request_id), None)
    if llm_request is None and session.llm_requests:
        llm_request = session.llm_requests[-1]
    if llm_request is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No LLM requests in session")

    # Find matching response
    llm_response = next((r for r in session.llm_responses if r.request_id == llm_request.id), None)
    if llm_response is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No LLM response found for request")

    # Get chunks
    chunks = []
    if session.vector_queries:
        query = session.vector_queries[-1]
        chunks = list(query.results)
    if not chunks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No chunks to analyze")

    # Build LLM caller
    from vectorlens.pipeline import _make_llm_caller
    from vectorlens.attribution.perturbation import PerturbationAttributor
    from vectorlens.detection.hallucination import HallucinationDetector
    from vectorlens.types import AttributionResult

    llm_caller = _make_llm_caller(llm_request.provider, llm_request.model)
    if llm_caller is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Provider '{llm_request.provider}' not supported for deep analysis (need OPENAI_API_KEY or ANTHROPIC_API_KEY)"
        )

    # Run detection + LIME
    detector = HallucinationDetector()
    output_tokens = detector.detect(llm_response.output_text, chunks)
    grounded = sum(1 for t in output_tokens if not t.is_hallucinated)
    overall_groundedness = grounded / len(output_tokens) if output_tokens else 1.0

    attributor = PerturbationAttributor(llm_caller)
    chunks = await attributor.compute_lime(llm_request.messages, chunks, llm_response.output_text)

    # Mark hallucination-causing chunks
    for chunk in chunks:
        chunk.caused_hallucination = overall_groundedness < 0.5 and chunk.attribution_score > 0.6

    result = AttributionResult(
        session_id=session_id,
        request_id=llm_request.id,
        response_id=llm_response.id,
        chunks=chunks,
        output_tokens=output_tokens,
        overall_groundedness=overall_groundedness,
        hallucinated_spans=[],
    )
    bus.record_attribution(result)
    return AttributionData.from_result(result)


class EmbeddingPoint(BaseModel):
    """A single point in 2D/3D embedding space for visualization."""
    id: str
    text: str           # snippet for tooltip
    x: float
    y: float
    z: float = 0.0
    type: str           # "chunk" | "graphrag_community" | "cag_document" | "query" | "hallucination"
    attribution_score: float = 0.0
    caused_hallucination: bool = False
    label: str = ""     # community title, doc title, etc.


class EmbeddingResponse(BaseModel):
    points: list[EmbeddingPoint]
    explained_variance: list[float] = Field(default_factory=list)  # PCA quality [PC1, PC2, PC3]


@router.get("/sessions/{session_id}/embeddings", response_model=EmbeddingResponse)
async def get_session_embeddings(session_id: str) -> EmbeddingResponse:
    """Return PCA-projected 2D/3D coordinates for all retrieval units in a session.

    Computes embeddings on demand (not stored). Uses PCA (numpy only, no extra deps)
    to project from 384-dim sentence-transformer space to 3D.

    Points include: retrieved chunks, GraphRAG community units, CAG documents,
    the LLM query, and any hallucinated output sentences — all in the same space.
    """
    import numpy as np
    from vectorlens.attribution.perturbation import _get_model

    session = bus.get_session(session_id)
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Session {session_id} not found")

    texts: list[str] = []
    meta: list[dict] = []

    # 1. Vector DB chunks (from latest attribution, not raw query — has attribution scores)
    attribution = session.attributions[-1] if session.attributions else None
    if attribution:
        for chunk in attribution.chunks:
            chunk_type = chunk.metadata.get("type", "chunk") if chunk.metadata else "chunk"
            label = chunk.metadata.get("title", "") if chunk.metadata else ""
            texts.append(chunk.text[:500])
            meta.append({
                "id": chunk.chunk_id,
                "text": chunk.text[:120],
                "type": chunk_type,
                "attribution_score": chunk.attribution_score,
                "caused_hallucination": chunk.caused_hallucination,
                "label": label,
            })

    # 2. Query text
    if session.vector_queries:
        q = session.vector_queries[-1]
        if q.query_text:
            texts.append(q.query_text[:500])
            meta.append({
                "id": "query",
                "text": q.query_text[:120],
                "type": "query",
                "attribution_score": 0.0,
                "caused_hallucination": False,
                "label": "Query",
            })
    elif session.llm_requests:
        req = session.llm_requests[-1]
        user_msg = next((m.get("content", "") for m in reversed(req.messages) if m.get("role") == "user"), "")
        if user_msg:
            texts.append(user_msg[:500])
            meta.append({
                "id": "query",
                "text": user_msg[:120],
                "type": "query",
                "attribution_score": 0.0,
                "caused_hallucination": False,
                "label": "Query",
            })

    # 3. Hallucinated output sentences
    if attribution:
        hall_texts = set()
        for tok in attribution.output_tokens:
            if tok.is_hallucinated and tok.text.strip() and tok.text not in hall_texts:
                hall_texts.add(tok.text)
                texts.append(tok.text[:500])
                meta.append({
                    "id": f"hall_{tok.position}",
                    "text": tok.text[:120],
                    "type": "hallucination",
                    "attribution_score": tok.hallucination_score,
                    "caused_hallucination": True,
                    "label": "Hallucinated",
                })

    if not texts:
        return EmbeddingResponse(points=[], explained_variance=[])

    # Embed all texts
    try:
        model = _get_model()
        embeddings = model.encode(texts, convert_to_numpy=True)  # (N, 384)
    except Exception as e:
        logger.warning(f"Embedding failed for scatter: {e}")
        return EmbeddingResponse(points=[], explained_variance=[])

    # PCA to 3D (numpy only)
    n_components = min(3, len(texts), embeddings.shape[1])
    centered = embeddings - embeddings.mean(axis=0)
    try:
        _, s, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:n_components].T  # (N, n_components)
        total_var = float((s ** 2).sum())
        explained = [float(s[i] ** 2 / total_var) for i in range(n_components)] if total_var > 0 else []
    except Exception:
        projected = np.zeros((len(texts), n_components))
        explained = []

    # Normalize to [-1, 1] for consistent canvas rendering
    for col in range(n_components):
        col_range = projected[:, col].max() - projected[:, col].min()
        if col_range > 0:
            projected[:, col] = (projected[:, col] - projected[:, col].min()) / col_range * 2 - 1

    points = []
    for i, m in enumerate(meta):
        coords = projected[i].tolist()
        points.append(EmbeddingPoint(
            id=m["id"],
            text=m["text"],
            x=float(coords[0]) if len(coords) > 0 else 0.0,
            y=float(coords[1]) if len(coords) > 1 else 0.0,
            z=float(coords[2]) if len(coords) > 2 else 0.0,
            type=m["type"],
            attribution_score=m["attribution_score"],
            caused_hallucination=m["caused_hallucination"],
            label=m["label"],
        ))

    return EmbeddingResponse(points=points, explained_variance=explained)


@router.post("/sessions/new", response_model=SessionSummary)
async def create_session() -> SessionSummary:
    """Create a new session and set it as active."""
    session = bus.new_session()
    return SessionSummary.from_session(session)


@router.delete("/sessions", status_code=status.HTTP_204_NO_CONTENT)
async def clear_all_sessions() -> None:
    """Delete all sessions."""
    bus.clear_all_sessions()


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str) -> None:
    """Delete a session."""
    deleted = bus.delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
