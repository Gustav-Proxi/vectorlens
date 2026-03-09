# CLAUDE.md — VectorLens Developer Guide

## What This Is

**VectorLens** — zero-config RAG hallucination debugger. `import vectorlens; vectorlens.serve()` starts a local dashboard showing token-level attribution: which retrieved chunks caused each output sentence?

## Key Commands

- **Tests (no ML models)**: `python -m pytest tests/ -m "not integration"`
- **Tests (with models)**: `python -m pytest tests/ -m integration` (slow — loads sentence-transformers)
- **Build dashboard**: `cd dashboard && npm run build`
- **Dev with dashboard**: `cd dashboard && npm run dev` (in parallel with `python -m pytest -xvs`)
- **Install dev mode**: `pip install -e ".[all]"`

## Architecture Overview

**Flow**: Interceptor → SessionBus → Pipeline → Server API → React Dashboard

- **`interceptors/`**: Monkey-patches for OpenAI, Anthropic, ChromaDB, Pinecone, FAISS, Weaviate, Gemini, HuggingFace. Each follows `BaseInterceptor` pattern: `install()`, `uninstall()`, `is_installed()`.
- **`session_bus.py`**: In-process event stream. Interceptors publish events (vector queries, LLM requests/responses); pipeline subscribes.
- **`pipeline.py`**: Auto-attribution on background thread. On every `llm_response` event: detects hallucinations, computes chunk attributions, updates session.
- **`detection/`**: Sentence-transformers embedding + cosine similarity to detect hallucinated tokens.
- **`attribution/`**: Perturbation scoring — drops chunks, measures output divergence.
- **`server/api.py`**: FastAPI endpoints for session retrieval, event streaming via WebSocket.
- **`dashboard/`**: React + TypeScript UI. Communicates with server via `/api/...` (relative URLs, port-agnostic).

## Key Patterns

1. **All interceptors are install/uninstall pairs**: Never import the actual client library directly in tests — mock it. Mocking pattern in `tests/conftest.py`.
2. **Events are immutable**: `LLMResponseEvent`, `VectorQueryEvent` are dataclasses. Pipeline reads, updates via `bus.record_*()` methods.
3. **Attribution is non-blocking**: Runs in daemon thread. Main pipeline never waits. Long operations (perturbation passes) don't hang the user's code.
4. **Sentence-transformers model is lazy-loaded**: `_get_model()` singleton pattern. Loaded on first `HallucinationDetector()` instantiation. On macOS MPS, model must be warmed BEFORE `serve()` (see RAG/scripts/test_vectorlens.py).
5. **Dashboard must be rebuilt after frontend changes**: `npm run build` produces `dashboard/dist/`. Server serves this static folder at root. Hot reload works in dev mode only.
6. **Tests mock sentence-transformers**: Run `pytest -m integration` for real model tests (slow). Unit tests use dummy embeddings.

## Known Issues / Gotchas

- **BGE-M3 and multiprocessing models on macOS**: Must call `model.warmup()` before `vectorlens.serve()`. The model spawns subprocesses; MPS warmup needs to happen in main process. See `RAG/scripts/test_vectorlens.py::test_rag_with_bge()`.
- **CORS is localhost-only by design**: `allow_origins` is restricted to `127.0.0.1:7756`. If you need Vite dev server access add `http://localhost:5173` to the list in `server/app.py`. Do NOT use `"*"` — that allows any webpage to steal session data.
- **MAX_SESSIONS=200**: Oldest sessions are LRU-evicted when limit is hit. Increase in `session_bus.py` if needed, but watch memory.
- **Attribution pool is bounded**: `ThreadPoolExecutor(max_workers=3)` in `pipeline.py`. If you need faster attribution on many concurrent calls, increase max_workers.
- **Interceptors never raise**: All `bus.record_*()` calls are wrapped in try/except. If VectorLens silently stops recording, check DEBUG logs — the exception is logged there.
- **Vector DB scores are clamped [0,1]**: Done defensively in all interceptors. If you see attribution scores at exactly 0.0 or 1.0 for many chunks, the raw scores from your DB may be outside range.
- **Dashboard dist must be rebuilt**: If frontend changes but you don't run `npm run build`, server will serve stale assets. No automatic rebuild during dev.
- **Tests only mock sentence-transformers by default**: Real model tests require `-m integration` flag. This slows CI — recommend running in a separate matrix job.
- **Port 7756 must be free**: Server binds hard to 7756. No automatic fallback. If already in use, `serve()` will hang silently (add timeout logging in future).
- **Session data is in-memory**: When server restarts, all sessions are lost. This is by design — local dev debugger, not production. For persistence, add SQLite backing (TODO).

## Testing Strategy

- **Unit tests**: Mock all LLM clients and vector DBs. Fast, deterministic.
- **Integration tests**: Real sentence-transformers model, but still mock LLM API responses. Medium speed.
- **E2E tests** (manual): Real RAG pipeline (see `RAG/scripts/test_vectorlens.py`). Load your actual LLM client + vector DB. This is the real test.

## Extending VectorLens

### Adding a new LLM provider

1. Create `vectorlens/interceptors/myprovider_patch.py`
2. Inherit from `BaseInterceptor` (see `openai_patch.py` as template)
3. Implement `install()` (patch the client), `uninstall()`, `is_installed()`
4. Add to `_INTERCEPTORS` dict in `interceptors/__init__.py`
5. Test: `python -c "from vectorlens.interceptors import install_all; install_all()"`

### Adding a new vector DB

Same pattern as LLM providers. Example: `vectorlens/interceptors/chroma_patch.py`.

### Changing detection algorithm

Edit `vectorlens/detection/hallucination.py`. Keep the interface: `HallucinationDetector.detect(output_text: str, chunks: list[RetrievedChunk]) -> list[OutputToken]`. Tests in `tests/test_detection.py`.

## Deployment & CI

- **GitHub Actions**: Tests run on every push. Integration tests run on `main` only (slow).
- **PyPI**: Publish via `hatch build && twine upload dist/`.
- **Dashboard**: Published as part of wheel (static files in `vectorlens/server/static/`). No separate CDN.
