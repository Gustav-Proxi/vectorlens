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

- **`interceptors/`**: Monkey-patches for LLM clients and vector DBs. httpx transport layer (OpenAI, Anthropic, Gemini, Mistral), plus SDK-specific patches (ChromaDB, Pinecone, FAISS, Weaviate, HuggingFace). LangChain framework patch. pgvector SQL interceptor. Each follows `BaseInterceptor` pattern: `install()`, `uninstall()`, `is_installed()`.
- **`session_bus.py`**: In-process event stream with `contextvars.ContextVar` session isolation. Interceptors publish events; pipeline subscribes.
- **`pipeline.py`**: Auto-attribution on background thread with bounded `ThreadPoolExecutor` (max_workers=3, max_pending=50). Conditional trigger: only runs deep attribution when hallucinations detected.
- **`detection/`**: Sentence-transformers embedding + cosine similarity to detect hallucinated tokens.
- **`attribution/`**: Perturbation scoring (N+1 method) and LIME-style bounded perturbation (fixed K calls). Attention rollout for local HuggingFace models.
- **`server/app.py`**: FastAPI app with pure ASGI body size middleware, CORS (localhost-only), WebSocket Origin validation.
- **`server/api.py`**: REST endpoints for session retrieval, event streaming.
- **`dashboard/`**: React + TypeScript UI communicating with server via `/api/...` (relative URLs).

## Key Patterns

1. **httpx transport interception**: All LLM SDKs use httpx for HTTP. Patching `httpx.AsyncClient.send` + `httpx.Client.send` is SDK-version-agnostic.
2. **ContextVar session isolation**: Each asyncio task and thread gets its own session via `contextvars.ContextVar`. Pre-set session_id is respected by `_resolve_session()`.
3. **Bounded attribution queue**: `threading.Semaphore(50)` gates ThreadPoolExecutor.submit(). Tasks dropped (logged as DEBUG) when queue full.
4. **Conditional attribution**: Skip deep attribution if `hallucinated_count == 0`.
5. **Attribution fallback chain**: For hallucinated responses: try attention rollout (local HF models) → fall back to LIME perturbation (API models) → record basic scores.
6. **Events are immutable dataclasses**: `LLMResponseEvent`, `VectorQueryEvent`, `AttributionResult` are immutable. Pipeline reads, updates via `bus.record_*()` methods.
7. **Sentence-transformers model is lazy-loaded**: `_get_model()` singleton pattern in `detection/hallucination.py`. Loaded on first instantiation.
8. **Dashboard must be rebuilt after frontend changes**: `npm run build` produces `dashboard/dist/`. Server serves this static folder at root.
9. **Tests mock sentence-transformers by default**: Run `pytest -m integration` for real model tests (slow). Unit tests use dummy embeddings.

## Known Issues / Gotchas

- **Modern transformers require attn_implementation='eager'**: SDPA (default) returns `None` for attentions. Load models with `attn_implementation='eager'` and `output_attentions=True` for token heatmap to work.
- **ContextVar session isolation**: `_active_session_var` is a `ContextVar` — each asyncio task/thread gets its own session. Tests that call `asyncio.run()` create a fresh context; use `bus.all_sessions()` to find events rather than relying on the pre-created session.
- **Attribution queue is bounded**: `ThreadPoolExecutor(max_workers=3)` with `Semaphore(50)` pending limit. Under extreme load, tasks are dropped. Check DEBUG logs for "Attribution queue full" messages.
- **httpx_transport.py intercepts ALL httpx traffic**: If you have custom httpx clients pointing to local/private LLM endpoints, add the hostname to `_LLM_HOSTS` in `httpx_transport.py`.
- **ASGI body middleware is now pure ASGI**: Not `BaseHTTPMiddleware`. If adding new middleware, stack order matters: `RequestSizeLimitMiddleware` must come first (outermost).
- **BGE-M3 and multiprocessing models on macOS**: Must call `model.warmup()` before `vectorlens.serve()`. The model spawns subprocesses; MPS warmup needs to happen in main process. See `RAG/scripts/test_vectorlens.py::test_rag_with_bge()`.
- **CORS is localhost-only by design**: `allow_origins` is restricted to `127.0.0.1:7756`, `localhost:7756`. If you need Vite dev server access add `http://localhost:5173` to the list in `server/app.py`. Do NOT use `"*"` — that allows any webpage to steal session data.
- **MAX_SESSIONS=200**: Oldest sessions are LRU-evicted when limit is hit. Increase in `session_bus.py` if needed, but watch memory.
- **Interceptors never raise**: All `bus.record_*()` calls are wrapped in try/except. If VectorLens silently stops recording, check DEBUG logs — the exception is logged there.
- **Vector DB scores are clamped [0,1]**: Done defensively in all interceptors. If you see attribution scores at exactly 0.0 or 1.0 for many chunks, the raw scores from your DB may be outside range.
- **Dashboard dist must be rebuilt**: If frontend changes but you don't run `npm run build`, server will serve stale assets. No automatic rebuild during dev.
- **Tests only mock sentence-transformers by default**: Real model tests require `-m integration` flag. This slows CI — recommend running in a separate matrix job.
- **Port 7756 must be free**: Server binds hard to 7756. No automatic fallback. If already in use, `serve()` will hang silently (add timeout logging in future).
- **Session data is in-memory**: When server restarts, all sessions are lost. This is by design — local dev debugger, not production. For persistence, add SQLite backing (TODO).
- **Streaming responses now captured**: httpx transport detects `text/event-stream` and reconstructs full text after stream ends. Streaming token counts are approximate (estimated from word count, not from SSE metadata).
- **LangChain interceptor requires langchain installed**: `pip install langchain`. Silently skips if not available.
- **pgvector interceptor buffers rows**: SQLAlchemy result is fully fetched into memory. For very large result sets (1000+ rows), this adds memory overhead. The interceptor only activates for queries containing `<=>`, `<->`, or `<#>` operators.
- **Streaming token counts are approximate**: Streaming responses don't include exact token counts in SSE chunks. VectorLens estimates completion_tokens from word count. Prompt tokens unavailable for streamed responses.
- **Conversation DAG is opt-in**: Call `bus.start_conversation()` to get a `conversation_id`. The DAG is built from `parent_request_id` fields; without explicit linking, multi-turn looks like isolated calls.
- **ChromaDB EphemeralClient hangs**: Don't use `chromadb.Client(settings=Settings(is_persistent=False))` with VectorLens active. Pre-compute embeddings instead.
- **pgvector interceptor: call vectorlens.serve() first**: Patch AsyncSession/Session before importing from `sqlalchemy.ext.asyncio`.
- **GraphRAG interceptor silently skips if graphrag not installed**: No warning emitted. If GraphRAG queries aren't being captured, verify `pip install graphrag` and check `get_installed()`.
- **GraphRAG LocalSearch text_chunks all have score=1.0**: `build_context()` doesn't expose per-chunk similarity; all text units treated as equally retrieved.
- **GraphRAG GlobalSearch attribution uses semantic similarity, not perturbation**: Community reports are scored by cosine similarity to hallucinated sentences. This is Tier 1 (zero LLM calls). Tier 2 (reduce-stage perturbation) is not yet implemented.
- **Token boundary alignment**: Pass `add_special_tokens=True` explicitly to ensure BOS/EOS token indices match between tokenizer calls.
- **SQLAlchemy 2.x with asyncpg**: Use `text().bindparams(**params)` instead of `session.execute(query, params)`.

## Testing Strategy

- **Unit tests**: Mock all LLM clients and vector DBs. Fast, deterministic.
- **Integration tests**: Real sentence-transformers model, but still mock LLM API responses. Medium speed.
- **E2E tests** (manual): Real RAG pipeline (see `RAG/scripts/test_vectorlens.py`). Load your actual LLM client + vector DB. This is the real test.

## Extending VectorLens

### Adding a new LLM provider

1. If SDK uses httpx (OpenAI, Anthropic, Gemini, Mistral), httpx transport already covers it.
2. If SDK uses custom HTTP transport, create `vectorlens/interceptors/myprovider_patch.py`
3. Inherit from `BaseInterceptor` (see `openai_patch.py` as template)
4. Implement `install()` (patch the client), `uninstall()`, `is_installed()`
5. Add to `_INTERCEPTORS` dict in `interceptors/__init__.py`
6. Test: `python -c "from vectorlens.interceptors import install_all; install_all()"`

### Adding a new vector DB

Same pattern as LLM providers. Example: `vectorlens/interceptors/chroma_patch.py`.

### Changing detection algorithm

Edit `vectorlens/detection/hallucination.py`. Keep the interface: `HallucinationDetector.detect(output_text: str, chunks: list[RetrievedChunk]) -> list[OutputToken]`. Tests in `tests/test_detection.py`.

## Deployment & CI

- **GitHub Actions**: Tests run on every push. Integration tests run on `main` only (slow).
- **PyPI**: Publish via `hatch build && twine upload dist/`.
- **Dashboard**: Published as part of wheel (static files in `vectorlens/server/static/`). No separate CDN.
