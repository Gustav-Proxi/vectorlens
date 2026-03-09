# VectorLens Context

## Project Overview

**VectorLens** is a zero-config RAG debugger that detects hallucinations and attributes them to retrieved chunks in real-time. Three lines of code (`import vectorlens; vectorlens.serve()`) start a local dashboard showing which chunks caused which hallucinated sentences, with attribution scores (0–100%) indicating influence.

**Problem solved**: RAG debugging is painful—when an LLM hallucinates, you're left guessing which chunk caused it. VectorLens automatically detects hallucinations via semantic similarity and shows chunk-level attribution without configuration.

**Current status** (v0.1.0): Shipped to PyPI, 117 tests passing, GitHub live.

## Tech Stack

- **Backend**: Python 3.11+, FastAPI, uvicorn, aiosqlite
- **ML**: sentence-transformers (all-MiniLM-L6-v2, 22MB, CPU-only)
- **Frontend**: React 18 + TypeScript, Tailwind CSS, Vite
- **Integrations**: OpenAI, Anthropic, Google Gemini, ChromaDB, Pinecone, FAISS, Weaviate, HuggingFace Transformers
- **Testing**: pytest with markers (`@pytest.mark.integration` for slow tests)

## Architecture

**Data Flow**: Interceptor → SessionBus (event stream) → Pipeline (attribution) → FastAPI → WebSocket → React Dashboard

- **Interceptors** (`vectorlens/interceptors/`): Monkey-patches LLM clients (OpenAI, Anthropic, Gemini) and vector DBs (ChromaDB, Pinecone, FAISS, Weaviate). Each patch captures requests/responses and publishes events to the bus.
- **SessionBus** (`session_bus.py`): Thread-safe in-process event stream. No network calls, no external services. Stores up to 200 sessions with LRU eviction.
- **Pipeline** (`pipeline.py`): Background thread with bounded ThreadPoolExecutor (max_workers=3). Subscribes to `llm_response` events; runs hallucination detection + chunk attribution asynchronously (non-blocking).
- **Detection** (`detection/hallucination.py`): Embeds sentences and chunks using sentence-transformers, computes cosine similarity, flags hallucinated tokens if max similarity < 0.4 (conservative threshold). Returns `list[OutputToken]` with `chunk_attributions` dict.
- **Attribution** (`attribution/perturbation.py`): Optional expensive path—drops chunks and re-runs LLM to measure output divergence (N+1 LLM calls, disabled by default).
- **Server** (`server/app.py`, `server/api.py`): FastAPI with WebSocket for real-time event streaming. REST endpoints for session CRUD, static React dashboard serving.

## File Structure

```
vectorlens/
├── __init__.py              # Public API: serve(), stop(), new_session(), get_session_url()
├── types.py                 # Dataclasses: EventType, Session, LLMRequestEvent, AttributionResult, etc.
├── session_bus.py           # SessionBus singleton: thread-safe event stream + session manager
├── pipeline.py              # Auto-attribution pipeline (subscribes to llm_response)
├── cli.py                   # CLI entry point
├── interceptors/
│   ├── base.py              # BaseInterceptor abstract class
│   ├── openai_patch.py      # Patches openai.resources.chat.completions.Completions.create
│   ├── anthropic_patch.py   # Patches anthropic.resources.messages.Messages.create
│   ├── gemini_patch.py      # Patches google.generativeai.GenerativeModel.generate_content (SDK v1 & v2)
│   ├── chroma_patch.py      # Patches chromadb.api.models.Collection.query
│   ├── pinecone_patch.py    # Patches pinecone.Index.query
│   ├── faiss_patch.py       # Wraps faiss.Index.search
│   ├── weaviate_patch.py    # Patches weaviate.Client query methods
│   ├── transformers_patch.py# Patches huggingface pipeline inference
│   └── __init__.py          # Registry: install_all(), uninstall_all(), get_installed()
├── detection/
│   ├── hallucination.py     # HallucinationDetector: embed sentences, cosine similarity, detect tokens
│   └── __init__.py
├── attribution/
│   ├── perturbation.py      # PerturbationAttributor: drop chunks, re-run LLM, measure divergence
│   ├── attention.py         # Token-level attention extraction (experimental, disconnected)
│   └── __init__.py
└── server/
    ├── app.py               # FastAPI app, WebSocket endpoint, static serving, CORS, request size limits
    ├── api.py               # REST endpoints (GET /status, /sessions, /sessions/{id}, etc.)
    └── __init__.py

tests/                        # 117 tests total
├── test_detection.py        # HallucinationDetector unit + integration tests
├── test_attribution.py      # PerturbationAttributor tests
├── test_pipeline.py         # Auto-attribution pipeline tests
├── test_interceptors.py     # Interceptor install/uninstall tests (mocked clients)
├── test_server.py           # FastAPI endpoint tests
├── test_types.py            # SessionBus, Session, event serialization tests
├── test_integration.py      # Full end-to-end with mocked LLM + real sentence-transformers
└── conftest.py              # pytest fixtures, mocked interceptor clients

dashboard/                    # React + TypeScript + Tailwind
├── src/
│   ├── App.tsx              # Main app component, session routing
│   ├── components/
│   │   ├── SessionList.tsx  # Left sidebar: session history, active indicator
│   │   ├── OutputHighlighter.tsx # Center: LLM output with red hallucinated spans
│   │   ├── ChunkCard.tsx    # Right: retrieved chunks, attribution %, similarity scores
│   │   └── AttributionView.tsx # Full attribution details view
│   ├── hooks/
│   │   └── useSession.ts    # Custom hook: fetch session via /api/sessions/{id}
│   ├── lib/
│   │   ├── api.ts           # API client (relative URLs, port-agnostic)
│   │   └── storage.ts       # localStorage for session persistence
│   ├── main.tsx
│   └── index.css            # Tailwind globals
├── package.json
├── vite.config.ts           # Vite dev server config
└── tsconfig.json

CLAUDE.md                     # Developer guide: commands, architecture, known gotchas
devnotes.md                   # Gap analysis: token-level attribution, streaming, LangChain integration
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
- All 8 LLM providers and 8 vector DB integrations fully patched
- Sentence-level hallucination detection (cosine similarity, threshold 0.4)
- Real-time WebSocket updates to React dashboard
- Session persistence (in-memory, LocalStorage for UI history)
- Attribution scores via normalized similarity weights
- Cost calculation (OpenAI, Anthropic, Gemini)
- 117 tests (unit + integration)
- Graceful error handling (all errors logged, never crash)

### Known Gaps
- **Sentence-level only** (not token-level): MVP uses sentence-transformers; true token-level requires attention weight extraction from local models only (OpenAI/Anthropic API models don't expose attention)
- **No streaming support**: Fully streamed outputs (`stream=True`) are ignored; chat UIs universally use streaming
- **No LangChain integration**: Only raw LLM clients patched; LCEL pipelines lack visibility into intermediate logic
- **pgvector requires manual adapter**: No native SQL DB patching (custom DB support via manual event API)
- **WebSocket no auth**: Assumes localhost-only; CORS restricted to `127.0.0.1:7756`, `localhost:5173` (Vite dev)
- **Session loss on restart**: In-memory only, no SQLite backing (TODO)
- **Port hard-coded to 7756**: No automatic fallback if busy (TODO)
- **12 GitHub issues open** (see GITHUB_ISSUES.md)

### Performance Notes
- Embedding: ~50ms per sentence (CPU)
- Attribution pipeline: Non-blocking, background thread
- Dashboard: ~100ms WebSocket latency
- Memory: ~300MB for model + session data
- Model: ~100MB download on first use (no progress bar, silent)

## Architecture Decisions

1. **Monkey-patching over wrappers**: Zero changes to user code; interceptors patch client methods directly.
2. **In-process event bus**: No network overhead, perfect for local debugging; doesn't scale to distributed systems.
3. **Background attribution**: Bounded ThreadPoolExecutor (max_workers=3) prevents thread explosion; main code never waits.
4. **Lazy-loaded models**: sentence-transformers loads on first detection call, not on import; saves upfront cost if detection never runs.
5. **Sentence-level baseline**: Simpler than token-level; works well for catching major hallucinations; token-level is expensive and requires local LLM access.
6. **LRU session eviction**: MAX_SESSIONS=200 prevents unbounded memory growth during long debugging sessions.
7. **WebSocket over HTTP polling**: Real-time updates with lower latency and bandwidth; still single-threaded server (uvicorn).

## Testing Strategy

- **Unit tests** (fast): Mock sentence-transformers, LLM clients, vector DBs. ~2 sec runtime.
- **Integration tests** (slow): Real sentence-transformers model, mocked LLM API responses. ~30 sec runtime.
- **E2E tests** (manual): Real RAG pipeline in `/Users/vaishak/Downloads/projects/RAG/scripts/test_vectorlens.py`. Exercises real OpenAI/Anthropic clients and vector DBs.

CI runs unit tests on every push; integration tests on main only.
