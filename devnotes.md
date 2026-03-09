# VectorLens: Development Notes & Gap Analysis

The initial scaffold of VectorLens lays an incredibly strong foundation:
- The in-process `session_bus` pattern is excellent.
- The `SentenceTransformer` hallucination detection logic is clean.
- The concept of perturbation attribution (dropping chunks and measuring output divergence) is a brilliant way to handle black-box RAG grounding.

However, to elevate this from a proof-of-concept to a production-grade developer tool (true to the "local-first, invisible" design philosophy), it currently lacks several critical features and robust edge-case handling.

---

## 🛑 What VectorLens Currently Lacks (Combined Gap Analysis)

### 1. Granularity & Accuracy Gaps
*   **Token-Level Attribution:** Current MVP relies on SentenceTransformer which limits detection to *sentence-level*. True token-level attribution (linking a specific generated word back to a document) requires extracting attention weights from local HF models. (Note: partially implemented in `attention.py` but completely disconnected from OpenAI/Anthropic API models).
*   **Brittle "Perturbation Attribution":** The perturbation approach (dropping a chunk and re-running the LLM) is implemented, but the `_remove_chunk_from_messages()` logic currently relies on naive substring replacement. If LangChain or a developer heavily formats the chunk (e.g., truncating it or injecting complex meta-tags), the silent replacement fails and attribution breaks.
*   **Perturbation Cost Overhead:** Expensive perturbation is not auto-triggered because doing N+1 LLM calls (where N is the number of chunks) is too expensive and slow for production use. We need an intelligent fallback (e.g., only trigger on suspected hallucinated responses).

### 2. Integration & Ecosystem Gaps
*   **LangChain Interception:** Currently, only the underlying raw LLM clients (OpenAI/Anthropic) are patched. If a developer uses a LangChain `RetrievalQA` or LCEL pipeline, VectorLens lacks visibility into the intermediate chain logic and prompt formatting.
*   **pgvector Native Support:** Native SQL database embedding retrieval isn't auto-patched. Fetching chunks from `pgvector` currently requires manual 10-line adapter code rather than drop-in interception.

### 3. Execution & Performance Gaps
*   **Streaming Responses:** Fully streamed outputs (`stream=True`) are ignored/not captured. RAG Chat UIs almost exclusively use streaming, meaning VectorLens is currently blind to production-UI behavior.
*   **Synchronous Blocking in Interceptors:** The OpenAIPatch currently blocks the main thread to calculate costs and construct event payloads. Local telemetry must push immediately to an async queue to avoid introducing latency into the user's app.
*   **Heavy, Silent Model Downloads:** The SentenceTransformer model downloads ~100MB on first boot without a loading bar, which breaks the "invisible" developer experience.

### 4. UI Data & Observability Gaps
*   **No "Conversational Memory" (DAGs):** The event bus (`session_bus.py`) has no concept of a Conversation Tree. Multi-turn LangChain agents just look like isolated, erratic LLM calls on the dashboard. We need a way to link parent-child traces.

---

## 🗺️ Immediate Roadmap (Next Steps)

1.  **Fix Streaming Support:** Update `OpenAIInterceptor.install()` to handle generators.
2.  **Fix Perturbation Robustness:** Re-write the string replacement logic in `perturbation.py` to be regex/semantic-aware.
3.  **UI Data Binding:** Connect the React frontend over WebSockets to actually consume the `LLMResponseEvent` and `AttributionResult` payloads.

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
