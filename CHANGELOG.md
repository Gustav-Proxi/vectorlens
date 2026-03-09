# Changelog

All notable changes to VectorLens are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.2] — 2026-03-09

### Added
- **httpx transport-layer interceptor** (`interceptors/httpx_transport.py`) — patches `httpx.AsyncClient.send` and `httpx.Client.send` instead of SDK internals. SDK-version-agnostic. Covers OpenAI, Anthropic, Gemini, Mistral. Registered as `"httpx"` interceptor.
- **LIME-style bounded perturbation** (`attribution/perturbation.py`) — new `compute_lime()` method for fixed-cost attribution (K=7 LLM calls regardless of chunk count). Uses ridge regression on random masks to compute per-chunk importance weights. Replaces unbounded N+1 approach.
- **Smart/conditional attribution trigger** (`pipeline.py`) — only runs deep attribution when hallucinations are detected. Fully grounded responses skip deep attribution (~50ms vs ~500ms savings). For local HuggingFace models: uses attention rollout; for API models: uses LIME bounded perturbation.
- **Attention rollout wiring** (`attribution/attention.py`) — integrated into pipeline for zero-extra-cost attribution on local models.

### Fixed
- **ContextVar session isolation (HIGH)** — `_active_session_id` replaced with `contextvars.ContextVar`. Each asyncio task and thread gets its own session. Prevents cross-thread data bleed in concurrent RAG servers.
- **ASGI body middleware (CRITICAL)** — replaced `BaseHTTPMiddleware` with pure ASGI class wrapping `receive()`. POST/PUT/PATCH now work correctly. Previous implementation's `_stream` hack was silently ignored by Starlette.
- **Bounded attribution queue (HIGH)** — `threading.Semaphore(50)` gates task submission to `ThreadPoolExecutor`. Tasks dropped when >50 pending instead of growing unbounded. Prevents OOM under high load. Check DEBUG logs for "Attribution queue full" messages.
- **WebSocket Origin validation** — checked before `accept()`, 1008 (Policy Violation) on mismatch.
- **Attention.py zero-division clamp** — prevents NaN in edge cases.

## [0.1.1] — 2026-03-09

### Security
- **CORS restricted to localhost-only** — changed `allow_origins=["*"]` to explicit localhost allowlist; prevents cross-origin exfiltration of session data from malicious webpages
- **Interceptors: bus errors never propagate to user code** — all `bus.record_*()` calls wrapped in try/except; VectorLens bugs can no longer crash your RAG pipeline
- **Thread-safe install/uninstall** — per-instance `threading.Lock` prevents double-patching race condition when `install()` called concurrently
- **Score normalization** — ChromaDB, FAISS, Weaviate, Pinecone scores clamped to `[0.0, 1.0]`; previously negative scores possible with non-normalized embeddings
- **Request body size limit** — 1MB cap via Starlette middleware; prevents memory DoS via large POST bodies
- **Messages input validation** — `None` or non-list `messages` kwarg now coerced to `[]` safely

### Fixed
- **Thread explosion** — attribution pipeline now uses `ThreadPoolExecutor(max_workers=3)` instead of spawning a new thread per LLM response
- **Session memory unbounded** — `SessionBus` now enforces `MAX_SESSIONS=200` with LRU eviction of oldest sessions
- **Dashboard NaN crash** — `chunk.score.toFixed()` on null/NaN score no longer crashes React render; `safeFixed()` helper added
- **localStorage freeze** — sessions exceeding 4MB now stored as metadata-only summaries; prevents synchronous JSON.stringify blocking browser tab
- **Null-safe session ID** — `session.id.substring()` guarded against null
- **Polling reduced** — idle poll interval changed from 2s to 3s

## [0.1.0] — 2026-03-08

### Added

- **Zero-config interceptors** for major LLM providers:
  - OpenAI (GPT-4o, GPT-4 Turbo, GPT-3.5)
  - Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
  - Google Gemini (both SDK v1 and v2)
  - HuggingFace Transformers pipelines

- **Zero-config interceptors** for major vector databases:
  - ChromaDB (in-memory and persistent)
  - Pinecone (serverless indices)
  - FAISS (local dense search)
  - Weaviate (enterprise vector database)

- **Sentence-level hallucination detection** using `sentence-transformers/all-MiniLM-L6-v2`:
  - Cosine similarity-based semantic matching
  - Conservative 0.4 threshold (fewer false positives)
  - Detects hallucinated sentences in real time

- **Auto-attribution pipeline** running on background thread:
  - Non-blocking attribution computation
  - Runs after every LLM response
  - Computes chunk contributions via similarity scoring
  - Optional perturbation-based attribution (expensive, opt-in)

- **FastAPI server** with WebSocket support:
  - Session management (create, list, retrieve)
  - Real-time event streaming via WebSocket
  - REST API for session data retrieval
  - Automatic CORS handling

- **React + TypeScript dashboard**:
  - Session history with localStorage persistence (survives server restarts)
  - Live session indicator with event timestamps
  - Groundedness meter (0–100% scale with color coding)
  - LLM output panel with sentence-level highlighting
  - Retrieved chunks panel sorted by attribution
  - Token count & latency display
  - Estimated cost calculation for major LLM providers

- **Public Python API**:
  - `vectorlens.serve()` — start dashboard + install interceptors
  - `vectorlens.stop()` — shutdown + cleanup
  - `vectorlens.new_session()` — create fresh tracing session
  - `vectorlens.get_session_url()` — get dashboard URL for session
  - Manual event API via `session_bus.record_vector_query()`

- **Comprehensive test suite**:
  - 111 tests covering unit + integration scenarios
  - Pytest markers for fast (no-ML) and slow (with-ML) tests
  - Mock fixtures for all LLM clients and vector DBs
  - Integration tests with real `sentence-transformers` model

- **Custom vector database support**:
  - `pgvector` adapter pattern for PostgreSQL
  - Manual event API for unsupported DBs
  - Example integration code for Elasticsearch, Milvus

- **Developer-friendly documentation**:
  - README with architecture diagrams
  - CONTRIBUTING.md with step-by-step integration guides
  - CLAUDE.md with development gotchas and patterns
  - Type hints throughout codebase
  - 20+ code examples

### Technical Details

- **Architecture**: Event-driven monolith with thread-safe session bus
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, ~22MB, CPU-only)
- **Performance**: ~50ms per sentence embedding, non-blocking attribution pipeline
- **Minimum Dependencies**: FastAPI, uvicorn, sentence-transformers, numpy, aiosqlite, httpx
- **Python**: 3.11+ required
- **Dashboard**: React 18 + TypeScript + Tailwind CSS, Vite build
- **Packaging**: hatchling build backend, PyPI distribution

### Known Limitations

- Sentence-level detection (not token-level) — token-level coming in v0.2
- Perturbation attribution requires N additional LLM API calls (disabled by default)
- Session data in-memory only (SQLite backing planned)
- Port 7756 hardcoded (will be configurable in future)
- macOS MPS (Metal Performance Shaders) requires model warmup before `serve()`

[0.1.0]: https://github.com/YOUR_REPO/vectorlens/releases/tag/v0.1.0
