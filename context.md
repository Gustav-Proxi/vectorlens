# VectorLens Context

## Project Overview

**VectorLens** is a zero-config RAG debugger that detects hallucinations and attributes them to retrieved chunks in real-time. Three lines of code (`import vectorlens; vectorlens.serve()`) start a local dashboard showing which chunks caused which hallucinated sentences, with attribution scores (0–100%) indicating influence.

**Problem solved**: RAG debugging is painful—when an LLM hallucinates, you're left guessing which chunk caused it. VectorLens automatically detects hallucinations via semantic similarity and shows chunk-level attribution without configuration.

**Current status** (v0.1.5-dev): Shipped to PyPI. Added streaming capture, LangChain interceptor, conversation DAG, pgvector, token-level heatmap, GraphRAG support, CAG (Cache-Augmented Generation) support, and 2D embedding scatter visualization. All retrieval types (vector chunks, GraphRAG community reports, CAG documents, query, hallucinations) rendered in PCA-projected 2D space in the dashboard. 80 tests passing. GitHub live.

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, uvicorn, aiosqlite
- **ML**: sentence-transformers (all-MiniLM-L6-v2, 22MB, CPU-only)
- **Frontend**: React 18 + TypeScript, Tailwind CSS, Vite
- **Integrations**: OpenAI, Anthropic, Google Gemini, LangChain, ChromaDB, Pinecone, FAISS, Weaviate, pgvector, HuggingFace Transformers, Microsoft GraphRAG
- **Testing**: pytest with markers (`@pytest.mark.integration` for slow tests)

## Architecture

**Data Flow**: Interceptor → SessionBus (event stream) → Pipeline (attribution) → FastAPI → WebSocket → React Dashboard

- **Interceptors** (`vectorlens/interceptors/`): httpx transport-layer patches for OpenAI/Anthropic/Gemini/Mistral (SDK-version-agnostic), plus SDK-specific patches for vector DBs (ChromaDB, Pinecone, FAISS, Weaviate, HuggingFace). LangChain framework patch (BaseChatModel, BaseRetriever). pgvector SQL interceptor (SQLAlchemy AsyncSession, Session). GraphRAG context builder patches (LocalSearchMixedContext, GlobalSearchCommunityContext). Each patch captures requests/responses and publishes events to the bus.
- **SessionBus** (`session_bus.py`): Thread-safe in-process event stream with `contextvars.ContextVar` isolation. Each asyncio task and thread gets its own session. Stores up to 200 sessions with LRU eviction. Supports conversation DAG via `conversation_id` and `parent_request_id`.
- **Pipeline** (`pipeline.py`): Background thread with bounded ThreadPoolExecutor (max_workers=3, max_pending=50 via Semaphore). Subscribes to `llm_response` events; runs hallucination detection + conditional deep attribution asynchronously. Tries attention rollout (local HF) → LIME perturbation (API models) as attribution fallback.
- **Detection** (`detection/hallucination.py`): Embeds sentences and chunks using sentence-transformers, computes cosine similarity, flags hallucinated tokens if max similarity < 0.4. Returns `list[OutputToken]` with `chunk_attributions` dict. Model download shows progress on first use.
- **Attribution** (`attribution/perturbation.py`): Two methods: `compute()` (N+1 perturbation, expensive, backward compat) and `compute_lime()` (K=7 random masks, ridge regression, fixed cost). Robust chunk removal via 3-tier fallback (exact → 120-char prefix → first-sentence match). For local models: attention rollout (`attribution/attention.py`) with `compute_per_token()` extracts token-level attention weights for per-output-subword-token attribution; requires `attn_implementation='eager'` (not SDPA). For GraphRAG global search: `attribution/graphrag_attribution.py` — `CommunityAttributor` uses cosine similarity to attribute hallucinated sentences to community reports (zero extra LLM calls).
- **Server** (`server/app.py`, `server/api.py`): FastAPI with pure ASGI body size middleware (1MB limit), CORS (localhost-only), WebSocket Origin validation. REST endpoints for session CRUD, WebSocket for real-time event streaming.

## File Structure

```
vectorlens/
├── __init__.py              # Public API: serve(), stop(), new_session(), get_session_url()
├── types.py                 # Dataclasses: EventType, Session, LLMRequestEvent, AttributionResult, TokenHeatmapEntry, etc.
├── session_bus.py           # SessionBus singleton: ContextVar isolation, thread-safe event stream + session manager
├── pipeline.py              # Auto-attribution pipeline (subscribes to llm_response), bounded executor (3 workers, 50 pending)
├── cli.py                   # CLI entry point
├── interceptors/
│   ├── base.py              # BaseInterceptor abstract class
│   ├── httpx_transport.py   # Patches httpx.AsyncClient.send + httpx.Client.send (SDK-agnostic)
│   ├── openai_patch.py      # Patches openai.resources.chat.completions.Completions.create (fallback)
│   ├── anthropic_patch.py   # Patches anthropic.resources.messages.Messages.create (fallback)
│   ├── gemini_patch.py      # Patches google.generativeai.GenerativeModel.generate_content (fallback)
│   ├── langchain_patch.py   # Patches langchain BaseChatModel + BaseRetriever
│   ├── chroma_patch.py      # Patches chromadb.api.models.Collection.query
│   ├── pinecone_patch.py    # Patches pinecone.Index.query
│   ├── faiss_patch.py       # Wraps faiss.Index.search
│   ├── weaviate_patch.py    # Patches weaviate.Client query methods
│   ├── pgvector_patch.py    # Patches SQLAlchemy AsyncSession.execute + Session.execute
│   ├── transformers_patch.py# Patches huggingface pipeline inference
│   ├── graphrag_patch.py    # Patches GraphRAG LocalSearchMixedContext + GlobalSearchCommunityContext
│   └── __init__.py          # Registry: install_all(), uninstall_all(), get_installed()
├── detection/
│   ├── hallucination.py     # HallucinationDetector: embed sentences, cosine similarity, detect tokens
│   └── __init__.py
├── attribution/
│   ├── perturbation.py      # PerturbationAttributor: compute() (N+1), compute_lime() (K=7 fixed)
│   ├── attention.py         # AttentionAttributor: token-level attention extraction for local HF models
│   ├── graphrag_attribution.py # CommunityAttributor: semantic similarity attribution for GraphRAG global search
│   └── __init__.py
└── server/
    ├── app.py               # FastAPI app, RequestSizeLimitMiddleware (pure ASGI), WebSocket, CORS, static serving
    ├── api.py               # REST endpoints (GET /status, /sessions, /sessions/{id}, etc.)
    └── __init__.py

tests/                        # 80 tests total
├── test_detection.py        # HallucinationDetector unit + integration tests
├── test_attribution.py      # PerturbationAttributor + AttentionAttributor tests
├── test_pipeline.py         # Auto-attribution pipeline tests
├── test_interceptors.py     # Interceptor install/uninstall tests (mocked clients)
├── test_server.py           # FastAPI endpoint tests
├── test_types.py            # SessionBus, Session, event serialization tests
├── test_integration.py      # Full end-to-end with mocked LLM + real sentence-transformers
├── test_graphrag_attribution.py # CommunityAttributor, GraphRAGInterceptor, context parsing
└── conftest.py              # pytest fixtures, mocked interceptor clients

dashboard/                    # React + TypeScript + Tailwind
├── src/
│   ├── App.tsx              # Main app component, session routing
│   ├── components/
│   │   ├── SessionList.tsx  # Left sidebar: session history, active indicator
│   │   ├── OutputHighlighter.tsx # Center: LLM output with red hallucinated spans; toggle "Token heatmap" / "Sentence view"
│   │   ├── ChunkCard.tsx    # Right: retrieved chunks, attribution %, similarity scores
│   │   └── AttributionView.tsx # Full attribution details view
│   ├── hooks/
│   │   └── useSession.ts    # Custom hook: fetch session via /api/sessions/{id}
│   ├── lib/
│   │   ├── api.ts           # API client (relative URLs, port-agnostic)
│   │   └── storage.ts       # localStorage for session persistence (4MB limit)
│   ├── main.tsx
│   └── index.css            # Tailwind globals
├── package.json
├── vite.config.ts           # Vite dev server config
└── tsconfig.json

CHANGELOG.md                  # Version history: v0.1.3 (streaming, LangChain, conversation DAG, pgvector), v0.1.2 (httpx, LIME), v0.1.1 (security), v0.1.0
CLAUDE.md                     # Developer guide: commands, architecture, patterns, gotchas
context.md                    # This file — project overview, architecture, current state
devnotes.md                   # Gap analysis: token-level (future); architecture decisions (httpx, LIME, ContextVar, ASGI, Semaphore, streaming, LangChain, pgvector, conversation DAG)
README.md                     # Full documentation + examples
pyproject.toml               # Build config, dependencies, test markers
```

## How to Run

**Install for development**:
```bash
pip install -e ".[all]"
cd dashboard && npm install
```

**Start dashboard**:
```bash
python -c "import vectorlens; vectorlens.serve()"
# Opens http://127.0.0.1:7756
```

**Run tests**:
```bash
# Fast (no ML models):
python -m pytest tests/ -m "not integration"

# Slow (loads real sentence-transformers model):
python -m pytest tests/ -m integration

# Build and test with RAG project:
cd /Users/vaishak/Downloads/projects/RAG/scripts
python test_vectorlens.py  # Full E2E with real LLM clients + vector DB
```

**Build dashboard for production**:
```bash
cd dashboard && npm run build  # Produces dist/
# Server auto-serves from vectorlens/server/static/ in wheel
```

## Current State

### What's Working
- httpx transport interceptor covers all SDKs (OpenAI, Anthropic, Gemini, Mistral)
- Streaming capture: SSE chunks intercepted, full text reconstructed
- LangChain integration: BaseChatModel, BaseRetriever, LCEL pipelines
- pgvector native support: SQLAlchemy AsyncSession/Session with `<=>`, `<->`, `<#>` operators
- Conversation DAG: parent_request_id linking, chain_step labels, conversation_id grouping
- Robust chunk removal: 3-tier fallback (exact → 120-char prefix → first-sentence match)
- Model download progress: yellow warning + green confirmation on first use, silent on cache hit
- Sentence-level hallucination detection (cosine similarity, threshold 0.4)
- Smart/conditional attribution: skips deep work when fully grounded (~50ms vs ~500ms)
- LIME-style bounded perturbation (K=7, fixed cost regardless of chunk count)
- Token-level attention heatmap for local HuggingFace models (per-output-subword-token attribution, zero extra LLM calls)
- **GraphRAG support**: intercepts LocalSearch + GlobalSearch context builders; emits `GraphRAGContextEvent` with text chunks (local) or community units (global); semantic similarity attribution for community reports solves the "no discrete chunk" problem for global search; serialized as `RetrievedChunk` with `metadata["type"]="graphrag_community"` for dashboard display
- **CAG (Cache-Augmented Generation)**: `vectorlens.cag_session(documents)` context manager; registers full document corpus; same cosine similarity attribution as GraphRAG global; documents appear as `cag_document` type in scatter. Zero retrieval step — works with large-context models (Gemini 1.5, Claude, GPT-4o)
- **Embedding scatter visualization**: `GET /sessions/{id}/embeddings` endpoint runs PCA on all retrieval units + query + hallucinated sentences, returns 2D/3D coordinates. Dashboard "Embedding Space" tab shows SVG scatter — chunks/communities/CAG docs/query/hallucinations all in one semantic space. Points sized by attribution score, colored by type, hover tooltip shows text snippet
- Real-time WebSocket updates to React dashboard
- Session persistence (in-memory, LocalStorage for UI history)
- Attribution scores via normalized similarity weights
- Cost calculation (OpenAI, Anthropic, Gemini)
- ContextVar session isolation (no cross-thread data bleed)
- Pure ASGI body middleware (POST/PUT/PATCH now work correctly, 1MB limit)
- Bounded attribution queue (max_pending=50, tasks dropped when exhausted)
- WebSocket Origin validation (1008 on mismatch)
- Token heatmap with per-token attribution weights (TokenHeatmapEntry dataclass)
- OOM guard: skip token heatmap for sequences > 4096 tokens
- 67 tests (unit + integration)
- Graceful error handling (all errors logged, never crash)

### Known Gaps
- **Attention requires eager mode**: Modern transformers default to SDPA attention, which returns `None`. Load local models with `attn_implementation='eager'` to expose attention weights for token heatmap.
- **WebSocket no auth**: Assumes localhost-only; CORS restricted to `127.0.0.1:7756`, `localhost:5173` (Vite dev)
- **Session loss on restart**: In-memory only, no SQLite backing (TODO)
- **Port hard-coded to 7756**: No automatic fallback if busy (TODO)

### Performance Notes
- Embedding: ~50ms per sentence (CPU)
- Attribution pipeline: Non-blocking, background thread
- Dashboard: ~100ms WebSocket latency
- Memory: ~300MB for model + session data
- Model: ~100MB download on first use (progress bar on first use)

## Supported Integrations

| Category | Provider | Feature | Status |
|----------|----------|---------|--------|
| **LLM** | OpenAI | GPT-4o, GPT-4 Turbo, GPT-3.5 | ✓ |
| **LLM** | Anthropic | Claude 3.5 Sonnet, Claude 3 Opus | ✓ |
| **LLM** | Google Gemini | Gemini 2.0 Flash, SDK v1 & v2 | ✓ |
| **LLM** | Mistral | All models via httpx | ✓ |
| **LLM** | HuggingFace | Transformers pipelines | ✓ |
| **Framework** | LangChain | BaseChatModel, BaseRetriever, LCEL | ✓ |
| **Vector DB** | ChromaDB | in-memory & persistent | ✓ |
| **Vector DB** | Pinecone | serverless indices | ✓ |
| **Vector DB** | FAISS | local dense search | ✓ |
| **Vector DB** | Weaviate | enterprise vector DB | ✓ |
| **Vector DB** | pgvector | PostgreSQL + SQLAlchemy | ✓ |
| **Vector DB** | Custom | via manual event API | ✓ |
| **GraphRAG** | Microsoft GraphRAG | LocalSearch + GlobalSearch | ✓ |

## Architecture Decisions

1. **httpx transport interception over SDK patching**: Zero SDK version maintenance. All major LLM SDKs use httpx; patching at transport layer is unbreakable.
2. **ContextVar session isolation**: Each asyncio task and thread gets its own session automatically. Pre-set session_id is respected. Prevents cross-thread data bleed in concurrent servers.
3. **Pure ASGI middleware over BaseHTTPMiddleware**: Starlette's receive() is the true interface. Pure ASGI wrapping is reliable; BaseHTTPMiddleware hacks don't work.
4. **Bounded attribution queue (Semaphore)**: ThreadPoolExecutor uses unbounded queue; tasks dropped (logged) when >50 pending. Prevents OOM under extreme load.
5. **Conditional attribution trigger**: Only run deep work when hallucinations detected. Saves ~50ms on grounded responses.
6. **LIME perturbation (K fixed) over N+1 perturbation**: Fixed cost (K=7 calls) regardless of chunk count. Approximate but practical.
7. **Attention rollout for local models**: Zero extra LLM calls for token-level attribution on HuggingFace models.
8. **In-process event bus**: No network overhead, perfect for local debugging; doesn't scale to distributed systems.
9. **Lazy-loaded models**: sentence-transformers loads on first detection call, not on import; saves upfront cost if detection never runs.
10. **LRU session eviction**: MAX_SESSIONS=200 prevents unbounded memory growth during long debugging sessions.
11. **WebSocket over HTTP polling**: Real-time updates with lower latency and bandwidth; still single-threaded server (uvicorn).
12. **Streaming capture at httpx layer**: SSE chunks intercepted transparently; full text reconstructed after stream ends. Works across all SDKs using httpx.
13. **LangChain framework patch**: Patches BaseChatModel._generate/_agenerate and BaseRetriever.invoke/ainvoke for zero-config LCEL integration. Falls back to httpx if unavailable.
14. **pgvector SQL interceptor**: Patches SQLAlchemy execute methods; detects vector operators (`<=>`, `<->`, `<#>`); buffers rows into VectorQueryEvent. Caller still iterates normally.
15. **Conversation DAG via parent_request_id**: Links child LLM calls to parent requests. `bus.start_conversation()` creates conversation_id for grouping. `chain_step` labels role (e.g., "agent", "tool_use").
16. **3-tier chunk removal fallback**: Handles truncated/reformatted chunks in perturbation. Exact match → 120-char prefix → first-sentence fallback.
17. **GraphRAG community attribution via semantic similarity**: GraphRAG global search has no discrete retrieved chunks — only synthesized community reports. Attributing hallucinations by cosine similarity (hallucinated sentence vs. community text) solves this without extra LLM calls. Community with highest similarity to the hallucinated content most likely "inspired" the distortion. Zero extra API calls; uses the sentence-transformers model already loaded for detection.

## Testing Strategy

- **Unit tests** (fast): Mock sentence-transformers, LLM clients, vector DBs. ~2 sec runtime.
- **Integration tests** (slow): Real sentence-transformers model, mocked LLM API responses. ~30 sec runtime.
- **E2E tests** (manual): Real RAG pipeline in `/Users/vaishak/Downloads/projects/RAG/scripts/test_vectorlens.py`. Exercises real OpenAI/Anthropic clients and vector DBs.

CI runs unit tests on every push; integration tests on main only.
