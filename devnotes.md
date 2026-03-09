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
