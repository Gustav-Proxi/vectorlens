# VectorLens: Development Notes & Gap Analysis

The initial scaffold of VectorLens lays an incredibly strong foundation:
- The in-process `session_bus` pattern is excellent.
- The `SentenceTransformer` hallucination detection logic is clean.
- The concept of perturbation attribution (dropping chunks and measuring output divergence) is a brilliant way to handle black-box RAG grounding.

However, to elevate this from a proof-of-concept to a production-grade developer tool (true to the "local-first, invisible" design philosophy), it currently lacks several critical features and robust edge-case handling.

---

## Architecture Changes — 2026-03-09

### New: httpx transport interceptor

**Why**: SDK method patching breaks on internal SDK refactors. httpx is the stable transport layer — all major LLM SDKs (OpenAI, Anthropic, Gemini, Mistral) use it for HTTP.

**How**: Patches `httpx.AsyncClient.send` and `httpx.Client.send` at the transport layer. Detects LLM hosts by hostname; only intercepts generation endpoints (`/chat/completions`, `/messages`, `/generateContent`). Works with both sync and async clients.

**Tradeoff**: Slightly harder to test (need to mock httpx request/response objects), but unbreakable across SDK version changes. Zero maintenance when SDKs refactor their internal APIs.

**Files**:
- `vectorlens/interceptors/httpx_transport.py` (406 lines)
- Registered in `interceptors/__init__.py` as `"httpx"` interceptor

### New: LIME-style bounded perturbation

**Why**: N+1 LLM calls for N chunks is unbounded cost. 30 chunks = 30 API calls (~$5). LIME provides fixed-cost approximate attribution.

**How**: `compute_lime(n_samples=7)` generates K random binary masks over chunks, calls LLM for each mask, measures semantic similarity between masked output and original. Ridge regression fits mask vectors → similarity scores; coefficients ≈ per-chunk importance. Cost: exactly K calls, not N.

**Tradeoff**: Approximate (not exact per-chunk attribution like N+1 method), but fixed cost regardless of chunk count. Ridge regression may overfit on small K, but K=7 works well in practice.

**Files**:
- `vectorlens/attribution/perturbation.py::PerturbationAttributor.compute_lime()` (96 lines)
- `compute()` (N+1 method) preserved for backward compat and expensive analyses

### New: Conditional attribution trigger

**Why**: Running full attribution on every response wastes CPU when response is fully grounded. Skip deep work when `hallucinated_count == 0`.

**How**: After shallow detection, check hallucinated spans. If zero, record attribution result with basic scores and return. Skip expensive perturbation or attention rollout.

**Cost savings**: ~50ms → ~500ms on responses with no hallucinations.

**Files**:
- `vectorlens/pipeline.py::_run_attribution()` (lines 155–168)

### New: Attention rollout wiring

**Why**: For local HuggingFace models, extract token-level attention weights without extra LLM calls.

**How**: `_try_get_hf_model_and_tokenizer()` checks if response came from local HF model; if so, runs `AttentionAttributor.compute()` to extract attention scores per token.

**Files**:
- `vectorlens/pipeline.py::_run_attribution()` (lines 170–195)
- `vectorlens/attribution/attention.py` (existing, now wired)

---

## 🛑 What VectorLens Currently Lacks (Combined Gap Analysis)

### 1. Granularity & Accuracy Gaps
*   **Token-Level Attribution:** Current MVP relies on SentenceTransformer which limits detection to *sentence-level*. True token-level attribution (linking a specific generated word back to a document) requires extracting attention weights from local HF models. (Note: partially implemented in `attention.py` but completely disconnected from OpenAI/Anthropic API models).
*   **Brittle "Perturbation Attribution":** The perturbation approach (dropping a chunk and re-running the LLM) is implemented, but the `_remove_chunk_from_messages()` logic currently relies on naive substring replacement. If LangChain or a developer heavily formats the chunk (e.g., truncating it or injecting complex meta-tags), the silent replacement fails and attribution breaks.
*   **Perturbation Cost Overhead (FIXED):** Expensive perturbation is no longer auto-triggered (v0.1.0 issue). We now use intelligent conditional triggering + LIME bounded perturbation for fixed cost.

### 2. Integration & Ecosystem Gaps
*   **LangChain Interception:** Currently, only the underlying raw LLM clients (OpenAI/Anthropic) are patched. If a developer uses a LangChain `RetrievalQA` or LCEL pipeline, VectorLens lacks visibility into the intermediate chain logic and prompt formatting.
*   **pgvector Native Support:** Native SQL database embedding retrieval isn't auto-patched. Fetching chunks from `pgvector` currently requires manual 10-line adapter code rather than drop-in interception.

### 3. Execution & Performance Gaps
*   **Streaming Responses:** Fully streamed outputs (`stream=True`) are ignored/not captured. RAG Chat UIs almost exclusively use streaming, meaning VectorLens is currently blind to production-UI behavior.
*   **Synchronous Blocking in Interceptors (IMPROVED):** httpx transport now records events asynchronously via thread-safe callbacks. Old SDK-level patches may still block; but httpx layer is safe.
*   **Heavy, Silent Model Downloads:** The SentenceTransformer model downloads ~100MB on first boot without a loading bar, which breaks the "invisible" developer experience.

### 4. UI Data & Observability Gaps
*   **No "Conversational Memory" (DAGs):** The event bus (`session_bus.py`) has no concept of a Conversation Tree. Multi-turn LangChain agents just look like isolated, erratic LLM calls on the dashboard. We need a way to link parent-child traces.

---

## Bug Fixes — 2026-03-09

### Bug: ASGI body middleware broke POST

**Symptom**: Every POST returned 422 Unprocessable Entity after chunked-encoding fix in v0.1.1

**Root cause**: Starlette uses ASGI `receive()`, not `_stream`; `BaseHTTPMiddleware` re-injection via `_stream` was silently ignored

**Fix**: Pure ASGI class `RequestSizeLimitMiddleware` wrapping `receive()` callable with `buffered_receive()` closure. Handles both Content-Length and Transfer-Encoding: chunked correctly.

**Files**:
- `vectorlens/server/app.py::RequestSizeLimitMiddleware` (class, lines 33–84)

### Bug: ContextVar isolation

**Issue**: Attribution threads (running in their own context) don't accidentally create a new session when calling `record_attribution()` because `_resolve_session()` was added.

**Implementation**:
- `_active_session_var: ContextVar` (lines 42–44)
- `_resolve_session()` method (lines 133–139) — respects pre-set session_id, falls back to context session
- All `record_*()` methods call `_resolve_session()` instead of hardcoding context session

**Files**:
- `vectorlens/session_bus.py` (full rewrite of session isolation)

### Bug: Unbounded attribution queue

**Symptom**: High-throughput LLM calls (100+ per second) silently grow the executor's internal `SimpleQueue` → OOM crash

**Root cause**: `ThreadPoolExecutor.submit()` uses unbounded `queue.SimpleQueue`

**Fix**: `threading.Semaphore(50)` as a non-blocking gate. If queue has >50 pending tasks, drop the new one. Check DEBUG logs for "Attribution queue full" messages.

**Files**:
- `vectorlens/pipeline.py::_pending_sem` (line 31) + `_on_llm_response()` (lines 46–62)

### Bug: WebSocket infinite loop edge case

**Symptom**: Half-closed socket states never raise `WebSocketDisconnect` → infinite CPU spin on `receive_text()`

**Fix**: Catch all exceptions (not just `WebSocketDisconnect`) and break loop. Log and continue gracefully.

**Files**:
- `vectorlens/server/app.py::websocket_endpoint()` (lines 219–226)

### Bug: Attention.py zero-division clamp

**Symptom**: Edge cases (empty embeddings, zero norms) → NaN scores

**Fix**: Clamp in `_cosine_similarity()` and downstream code

**Files**:
- `vectorlens/attribution/perturbation.py::_cosine_similarity()` (lines 34–41)

---

## Security Audit — 2026-03-09

Red-teamed across backend, interceptors, and frontend. All critical/high issues patched in v0.1.1.

### Bugs Fixed

**Bug: CORS wildcard allows cross-origin session theft**
- Symptom: Any webpage could `fetch('http://127.0.0.1:7756/api/sessions')` and read all LLM prompts/responses
- Root cause: `allow_origins=["*"]` + `allow_credentials=True` — valid for dev shortcut but exploitable
- Fix: `allow_origins=["http://127.0.0.1:7756", "http://localhost:7756"]`, `allow_credentials=False`
- File: `vectorlens/server/app.py`

**Bug: Thread explosion under load**
- Symptom: 1000 LLM calls → 1000 daemon threads, each loading sentence-transformers → OOM/slowdown
- Root cause: `threading.Thread(...).start()` per response in `pipeline.py`
- Fix: `ThreadPoolExecutor(max_workers=3)` — bounded pool, tasks queue when all workers busy
- File: `vectorlens/pipeline.py`

**Bug: Unbounded session memory**
- Symptom: `POST /api/sessions/new` loop → server OOM crash
- Root cause: `bus._sessions` dict has no eviction policy
- Fix: `MAX_SESSIONS=200` + LRU insertion-order eviction via `_session_order` list
- File: `vectorlens/session_bus.py`

**Bug: Interceptor exception propagates into user code**
- Symptom: VectorLens internal error could crash user's `client.chat.completions.create()` call
- Root cause: `bus.record_llm_request(event)` called before the original function with no guard
- Fix: All `bus.record_*()` calls wrapped in `try/except` with `_logger.debug()`
- File: All `vectorlens/interceptors/*_patch.py`

**Bug: Double-install race condition**
- Symptom: Concurrent `install()` from two threads → method patched twice (wrapper wraps wrapper)
- Root cause: `if self._installed: return` check not atomic
- Fix: `self._install_lock = threading.Lock()` in `__init__`, wrap `install()` / `uninstall()` bodies
- File: All `vectorlens/interceptors/*_patch.py`

**Bug: Negative similarity scores from vector DBs**
- Symptom: ChromaDB cosine distance > 1.0 → `score = 1 - distance` becomes negative
- Root cause: Non-normalized embeddings produce distances outside [0, 1]
- Fix: `score = max(0.0, min(1.0, ...))` clamp in all vector DB patches
- Files: `chroma_patch.py`, `faiss_patch.py`, `weaviate_patch.py`, `pinecone_patch.py`

**Bug: Dashboard crash on NaN score**
- Symptom: Malformed API response with `score: null` → `chunk.score.toFixed(3)` throws → blank screen
- Root cause: TypeScript type says `number` but runtime allows `null`
- Fix: `safeFixed(n, digits)` helper with `isFinite()` guard
- File: `dashboard/src/components/ChunkCard.tsx`

**Bug: Browser freeze on large sessions**
- Symptom: Session with 500+ chunks → `JSON.stringify(sessions)` blocks main thread → tab unresponsive
- Root cause: Synchronous localStorage write with no size check
- Fix: 4MB size limit; if exceeded, store metadata-only summaries; if still too large, skip
- File: `dashboard/src/lib/storage.ts`

### Known Remaining Issues

- **WebSocket no auth** — by design for local dev; any localhost process can subscribe to events. Document as "do not expose port 7756 to the network" (in SECURITY.md).
- **LLM prompts stored in plaintext** — sessions contain full prompt history. Do not use VectorLens with production data containing PII or secrets.
- **No rate limiting on API endpoints** — mitigated by localhost-only binding; low priority until multi-user mode is added.

---

## Issue #8: Token-Level Attribution — COMPLETED

**What was built:**
- `TokenHeatmapEntry` dataclass in `vectorlens/types.py` — per-token scores with text, `chunk_attributions` dict
- `token_heatmap: list[TokenHeatmapEntry]` field added to `AttributionResult`
- `AttentionAttributor.compute_per_token()` — runs full [prompt+output] sequence, returns per-output-subword-token chunk attribution
- `pipeline.py::_run_attribution()` now calls `compute_per_token()` when local HF model available
- Dashboard `OutputHighlighter.tsx` toggle between "Token heatmap" and "Sentence view"
- `attn_implementation='eager'` required (SDPA returns None)
- OOM guard: skip token heatmap for sequences > 4096 tokens
- `tokenizer.convert_tokens_to_string()` used (replaces manual marker stripping)
- `_get_char_to_token_mapping()` accepts explicit `add_special_tokens` param
- None entries in offset_mapping handled (fast tokenizers return None for special tokens)
- `groundedColor` minimum alpha 30/255 so zero-score tokens stay visible
- 67 tests passing; new tests in `tests/test_attention_attribution.py`

---

## Critical Gotchas — Issue #8 & Beyond

### Modern Transformers Defaults to SDPA

Modern HuggingFace transformers default to `attn_implementation='sdpa'` (flash attention), which optimizes for speed but **returns `None` for attentions**. For token heatmap:

```python
model = AutoModelForCausalLM.from_pretrained(
    "distilgpt2",
    attn_implementation='eager',  # REQUIRED — use original attention mechanism
    output_attentions=True         # Ensure attention weights are computed
)
```

Without `eager`, `AttentionAttributor.compute_per_token()` receives `None` and falls back gracefully. Document this prominently in token heatmap feature announcement.

### ChromaDB EphemeralClient Hangs with VectorLens Interceptor

**Symptom**: `chromadb.Client(settings=Settings(is_persistent=False))` (in-memory ephemeral) hangs indefinitely when VectorLens interceptor is active.

**Root cause**: ChromaDB's ephemeral client spins up SQLite in-memory DB. VectorLens patches ChromaDB's `Collection.query()` — if the patch calls async methods or blocks, the SQLite connection can deadlock.

**Workaround**: Pre-compute embeddings and pass vectors directly:
```python
# ✗ DON'T: chromadb.Client(...)
# ✓ DO:
from sentence_transformers import SentenceTransformer
embeddings = SentenceTransformer('...').encode(texts)
chromadb.query(embeddings=embeddings)  # Pass vectors explicitly
```

For persistent collections, the issue doesn't occur.

### pgvector Interceptor Requires AsyncSession Patch BEFORE Imports

**Symptom**: pgvector queries return no results; vectorlens logs show no `VectorQueryEvent`.

**Root cause**: SQLAlchemy 2.x loads `AsyncSession` at import time. VectorLens patches methods in `__init__`. If you import AsyncSession before calling `vectorlens.serve()`, the original (unpatched) class is used.

**Fix**: Always call `vectorlens.serve()` FIRST, before importing `sqlalchemy.ext.asyncio.AsyncSession`:

```python
import vectorlens
vectorlens.serve()  # Install all interceptors

# NOW import SQLAlchemy
from sqlalchemy.ext.asyncio import AsyncSession
```

Or explicitly pre-patch:
```python
from vectorlens.interceptors import PgVectorInterceptor
PgVectorInterceptor().install()
```

### Token Boundary Alignment: add_special_tokens Must Be Explicit

**Symptom**: Token indices don't align; special tokens (BOS/EOS) get skipped in attention heatmap.

**Root cause**: `tokenizer(text)` vs `tokenizer(text, add_special_tokens=True)` can produce different token counts. When computing attention weights vs offset_mapping, indices mismatch.

**Fix**: Pass `add_special_tokens=True` explicitly in both calls:

```python
tokens = tokenizer(prompt + output, add_special_tokens=True, return_offsets_mapping=True)
# Ensure _get_char_to_token_mapping() also receives add_special_tokens=True
```

The `_get_char_to_token_mapping()` function now accepts an explicit `add_special_tokens` parameter to enforce alignment.

### Use `text().bindparams(**params)` for SQLAlchemy 2.x with asyncpg

**Symptom**: `session.execute(query, params)` fails or behaves inconsistently with asyncpg driver in SQLAlchemy 2.x.

**Root cause**: SQLAlchemy 2.x separates compile-time params from bind-time params. asyncpg requires all params bound at compile time.

**Fix**: Use `text().bindparams()`:

```python
# ✗ DON'T:
result = await session.execute(text("SELECT * FROM table WHERE id = :id"), {"id": 42})

# ✓ DO:
from sqlalchemy import text
query = text("SELECT * FROM table WHERE id = :id").bindparams(id=42)
result = await session.execute(query)
```

---

---

## GraphRAG Support — 2026-03-12

### Problem: attribution for GlobalSearch has no discrete retrieval units

Standard RAG returns text chunks from a vector DB — easy to attribute. GraphRAG's GlobalSearch uses map-reduce over community reports (LLM-synthesized summaries). There's no chunk to remove and re-query. This was an open research problem.

**Solution: community reports as attribution units + semantic similarity scoring**

- Community reports ARE discrete — `GlobalSearchCommunityContext.build_context()` selects N of them
- For a hallucinated sentence H and community C: `cosine_sim(embed(H), embed(C.text))` is a meaningful attribution signal — the community that provided the most "relevant but distorted" context will be semantically closest to what the LLM hallucinated
- This requires zero extra LLM calls (reuses the sentence-transformers model already loaded)
- Attribution scores normalized to [0,1]; `caused_hallucination` flagged at > 0.5 post-normalization

**Implementation**:
- `interceptors/graphrag_patch.py`: patches `LocalSearchMixedContext.build_context()` and `GlobalSearchCommunityContext.build_context()`. Extracts text units (local) or community reports (global) and emits `GraphRAGContextEvent` to the bus.
- `attribution/graphrag_attribution.py`: `CommunityAttributor.attribute()` — cosine similarity over hallucinated sentences vs. community texts. Max-pooling across sentences.
- `pipeline.py::_run_graphrag_global_attribution()` — new branch for global search: uses `HallucinationDetector` with community texts as pseudo-chunks, then `CommunityAttributor` for attribution. Serialized as `RetrievedChunk` with `metadata["type"]="graphrag_community"` for API compatibility.
- `types.py`: `GraphRAGCommunityUnit`, `GraphRAGContextEvent`, `Session.graphrag_contexts`
- `session_bus.py`: `record_graphrag_context()`

**Interception strategy for LocalSearch**:
Source text units live in `context_data["sources"]` or `context_data["text_units"]` from `build_context()`. Flattened to `RetrievedChunk` objects → existing LIME pipeline works unchanged.

**Community report object format is version-dependent**: graphrag's `CommunityReport` dataclass uses `summary` or `full_content` field depending on version. `_community_unit_from_report()` handles dict, dataclass, and string via attribute priority chain.

**Context string parsing fallback**: If `context_data` dict doesn't contain community reports in an expected key, `_parse_community_sections()` parses the formatted context string. Supports `## Header` sections and `---` separator blocks.

### Gotchas

- **graphrag is not installed by default**: `GraphRAGInterceptor.install()` silently skips on `ImportError`. No warning emitted — user must `pip install graphrag`.
- **GlobalSearch makes concurrent map LLM calls**: httpx transport captures these as individual `LLMResponseEvent` entries. They appear as multiple LLM calls per query in the session. Expected behavior.
- **Community report text may be truncated**: GraphRAG may store summaries vs full_content — `_community_unit_from_report()` prefers `summary` for brevity in attribution UI.
- **Local search text_chunks get score=1.0**: GraphRAG doesn't expose per-chunk similarity scores from `build_context()`. All text units are treated as equally retrieved.
- **Pipeline priority**: `session.graphrag_contexts` takes priority over `session.vector_queries` in `_run_attribution()`. If a session has both (unusual), GraphRAG context wins.

### Future: Reduce-Stage Perturbation (Tier 2)

For even more accurate global search attribution:
1. Capture the reduce LLM call from httpx (messages contain all map responses)
2. Remove one community's section from the reduce messages
3. Re-run reduce LLM call, measure output change
4. High change → high attribution

This costs K extra LLM calls (one per candidate community) but is more accurate than similarity. Not implemented in v1 — similarity already solves the discrete-unit problem.

---

## 🗺️ Immediate Roadmap (Next Steps)

1. **GraphRAG dashboard display** — ChunkCard.tsx should render `metadata["type"]="graphrag_community"` differently (show community title, entity count, not just chunk text).
2. **Broadcast token heatmap feature** — update README, changelog, documentation.
3. **Streaming edge cases** — test token heatmap with streaming responses.
4. **Perturbation robustness** — investigate chunk removal failures with real RAG pipelines.
5. **GraphRAG Tier 2 attribution** — reduce-stage perturbation for higher accuracy global search attribution.
