# VectorLens GitHub Issues — Roadmap & Contribution Guide

---

## Issue 1: Add LangChain Interceptor

**Labels:** `good first issue`, `new-interceptor`, `enhancement`
**Milestone:** v0.2.0

### Description

Currently, VectorLens only patches low-level LLM clients (OpenAI, Anthropic, Gemini) and vector DBs directly. If a developer uses **LangChain's high-level abstractions** (e.g., `RetrievalQA`, LCEL chains, `ChatPromptTemplate`), VectorLens has no visibility into the prompt formatting, intermediate chain steps, or how chunks flow through the pipeline.

This issue is to add a **LangChain interceptor** that patches `langchain.chat_models.base.BaseChatModel._generate` and `_agenerate` methods. This allows VectorLens to capture:
- The final prompt *after* formatting (vs. the raw user message)
- Chain metadata (chain type, tool usage)
- Multi-step LLM calls within agent loops

**Why this matters**: LangChain is the most popular RAG framework. Without this, VectorLens is blind to the majority of production RAG systems.

**Reference files**:
- See `vectorlens/interceptors/openai_patch.py` for the exact pattern: `install()` → store original → wrap with `@functools.wraps()` → record event → call original → extract response → record event.
- Pattern also in `vectorlens/interceptors/anthropic_patch.py`.

### Acceptance Criteria
- [ ] New file `vectorlens/interceptors/langchain_patch.py` with `LangChainInterceptor` class
- [ ] Patches `BaseChatModel._generate()` and `_agenerate()` methods
- [ ] Records `LLMRequestEvent` (with formatted prompt) and `LLMResponseEvent` for each LangChain call
- [ ] Handles both sync and async invocations
- [ ] Added to `_INTERCEPTORS` dict in `vectorlens/interceptors/__init__.py`
- [ ] Unit tests in `tests/test_interceptors/test_langchain_patch.py` (mock LangChain, no real API calls)
- [ ] Tests verify request/response events are recorded correctly
- [ ] `install()` silently skips if LangChain not installed

### Implementation Notes
- LangChain uses internal `_generate()` for sync and `_agenerate()` for async — both must be patched
- Extract prompt text from the `messages` parameter (list of `BaseMessage` objects)
- Extract output from `LLMResult` returned by original method
- Token counts may be unavailable (LangChain abstracts them) — use None if missing
- Test with both old (`langchain`) and new (`langchain_core`) module layouts
- Follow the exact wrapper pattern from `openai_patch.py` for consistency

---

## Issue 2: Add Configurable Hallucination Threshold

**Labels:** `good first issue`, `enhancement`, `api`
**Milestone:** v0.2.0

### Description

The hallucination detection threshold is currently **hardcoded at 0.4** (cosine similarity) in `vectorlens/detection/hallucination.py`. This means a sentence must match a chunk with at least 40% semantic similarity to be considered "grounded."

While 0.4 is reasonable for general use, different RAG applications need different sensitivities:
- **High precision** (fewer false hallucinations flagged): 0.5+
- **High recall** (catch subtle hallucinations): 0.2-0.3
- **Legal/medical**: 0.6+ (conservative)

This issue is to expose the threshold as a **configurable parameter** in the public API:

```python
import vectorlens
vectorlens.serve(hallucination_threshold=0.4)  # Default
# Or per-session:
vectorlens.new_session(hallucination_threshold=0.5)
```

**Why this matters**: One-size-fits-all thresholds cause either alert fatigue (false positives) or missed hallucinations. Developers should tune for their domain.

**Reference files**:
- Threshold is at `vectorlens/detection/hallucination.py:40` (approximately)
- Public API in `vectorlens/__init__.py` — `serve()` and `new_session()` functions
- Threshold used in `HallucinationDetector.detect()` method

### Acceptance Criteria
- [ ] Add `hallucination_threshold: float = 0.4` parameter to `vectorlens.serve()`
- [ ] Add `hallucination_threshold: float = 0.4` parameter to `vectorlens.new_session()`
- [ ] Store threshold in `Session` dataclass (`vectorlens/types.py`)
- [ ] Pass threshold to `HallucinationDetector` at instantiation
- [ ] Expose threshold in API response (`GET /sessions/{id}`) so dashboard can show it
- [ ] Default to 0.4 (backward compatible)
- [ ] Validate range: 0.0 < threshold <= 1.0, raise `ValueError` if out of bounds
- [ ] Unit tests verify different thresholds produce different hallucination flags

### Implementation Notes
- Update `HallucinationDetector.__init__()` to accept `threshold` parameter
- Store on `self.threshold`, use in `detect()` method where similarity is compared
- Session-level threshold should override global default
- Document in README: explain threshold tuning for different domains
- Add to dashboard: show current threshold in UI (read-only for now)

---

## Issue 3: Export Session as JSON

**Labels:** `good first issue`, `dashboard`, `api`
**Milestone:** v0.2.0

### Description

Currently, VectorLens sessions live in the dashboard and browser localStorage. To share a hallucination report with teammates or file a bug, users must:
1. Screenshot the dashboard
2. Manually copy output + chunks
3. Paste into Slack/GitHub

This is painful and error-prone. The solution: add a **JSON export endpoint** and "Export" button.

**Why this matters**: Sharing reproducible hallucination cases is critical for debugging production RAG systems. A shareable JSON export enables async collaboration without screenshots.

**Reference files**:
- Session structure in `vectorlens/types.py` — `Session` dataclass
- API endpoints in `vectorlens/server/api.py` — `FastAPI` app definition
- Dashboard buttons in `dashboard/src/components/` (reference existing UI)

### Acceptance Criteria
- [ ] New API endpoint: `GET /api/sessions/{id}/export` returns full session as JSON
- [ ] JSON includes: request/response events, attributed chunks, groundedness score, timestamps
- [ ] Response formatted nicely (indented, readable)
- [ ] Add "Export" button to dashboard UI (next to session name or in top bar)
- [ ] Button triggers fetch to `/api/sessions/{id}/export`, downloads as `session_{id}.json`
- [ ] Include metadata: VectorLens version, timestamp, threshold used
- [ ] Test JSON can be re-imported (for future feature)

### Implementation Notes
- Endpoint should serialize `Session` object using Pydantic `model_dump()` or similar
- Include all LLMRequestEvent, LLMResponseEvent, AttributionResult payloads
- Keep JSON human-readable (for manual inspection in bug reports)
- File naming: use session ID in filename so exported files don't collide
- Dashboard button: add to SessionPanel header or as icon next to "Delete"
- No authentication/permissions (local-only tool)

---

## Issue 4: LiteLLM Interceptor

**Labels:** `enhancement`, `new-interceptor`, `high-value`
**Milestone:** v0.3.0

### Description

[LiteLLM](https://github.com/BerriAI/litellm) is a universal LLM proxy that abstracts 100+ providers (OpenAI, Anthropic, Cohere, Ollama, Bedrock, Azure, HuggingFace, etc.) under a single API.

Currently, VectorLens requires a separate interceptor for each provider. By patching **LiteLLM's `completion()` and `acompletion()` functions**, we get support for 100+ providers with a *single patch*.

This is high-leverage: one interceptor adds support for dozens of less-common models (Ollama local, Bedrock, MistralAI, Hugging Face Inference API, etc.).

**Why this matters**: Not all users run OpenAI/Anthropic. Supporting LiteLLM unblocks local model debugging (Ollama), enterprise deployments (Bedrock, Azure), and cost-conscious teams (Cohere).

**Reference files**:
- Study LiteLLM's public API: `litellm.completion(model="...", messages=[...], ...)`
- Template: copy structure from `vectorlens/interceptors/openai_patch.py`
- Interceptor registry: `vectorlens/interceptors/__init__.py`

### Acceptance Criteria
- [ ] New file `vectorlens/interceptors/litellm_patch.py` with `LiteLLMInterceptor`
- [ ] Patches `litellm.completion()` and `litellm.acompletion()`
- [ ] Records `LLMRequestEvent` and `LLMResponseEvent` with model, tokens, latency
- [ ] Handles streaming responses (if applicable) — at minimum, buffer full response on stream end
- [ ] Registered in `_INTERCEPTORS` dict
- [ ] Unit tests (mock LiteLLM, no real API calls)
- [ ] Integration test with real LiteLLM + local provider (e.g., mock provider)
- [ ] Silently skip if LiteLLM not installed

### Implementation Notes
- LiteLLM response structure differs slightly from native APIs — parse carefully
- `model` parameter may be fully qualified (e.g., `"gpt-4-0125-preview"` → extract base name)
- Token counts: use `completion_tokens` and `prompt_tokens` from response if available
- Streaming: capture the full buffered response when streaming ends
- Test with both `completion()` (simple) and `acompletion()` (async)
- Document: add LiteLLM to supported providers table in README

---

## Issue 5: Streaming Response Support

**Labels:** `enhancement`, `help wanted`, `high-priority`
**Milestone:** v0.2.0

### Description

Most RAG chatbots stream responses to users (streaming tokens as they're generated). However, VectorLens currently **ignores streamed outputs entirely** — it only captures complete responses from blocking calls.

This is a major gap: production RAG UIs almost always use streaming (`stream=True` in OpenAI/Anthropic). VectorLens is blind to 90% of real-world RAG interactions.

**Example problematic code**:
```python
# Streaming — currently NOT captured by VectorLens
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    stream=True  # ← VectorLens ignores this
)
for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

**Why this matters**: Streaming is non-negotiable for user experience. VectorLens must handle it to be useful in production.

**Reference files**:
- Current OpenAI patch: `vectorlens/interceptors/openai_patch.py` — see `_wrap_create()` method
- Same issue in: `anthropic_patch.py`, `gemini_patch.py`
- Response parsing logic in each patch

### Acceptance Criteria
- [ ] Detect `stream=True` or `stream=False` in all LLM interceptors (OpenAI, Anthropic, Gemini)
- [ ] For streamed responses: buffer the generator, collect all chunks, assemble final text
- [ ] Record `LLMResponseEvent` only after streaming completes (with full text + total tokens)
- [ ] Preserve latency measurement: start time when generator created, end time when stream ends
- [ ] For non-streamed: no behavior change (backward compatible)
- [ ] Tests verify both streaming and non-streaming capture responses correctly
- [ ] Integration test: real streaming call (with mocked LLM response generator)

### Implementation Notes
- Python generators: wrap with `itertools.tee()` or collect into list (careful: memory usage for large responses)
- Token counting: may not be available per-chunk; estimate from final text or use cumulative from response metadata
- Latency: start timer before yielding generator, end after `.close()` or iteration complete
- Handle edge cases: partial chunks, errors mid-stream, empty streams
- OpenAI: use `token_usage` from final streaming chunk if available
- Anthropic: track `message_start` and `message_delta` events in stream

---

## Issue 6: Perturbation Attribution UI — Deep Attribution Button

**Labels:** `enhancement`, `dashboard`, `ui`
**Milestone:** v0.3.0

### Description

VectorLens has a sophisticated **perturbation attribution** system (drop each chunk, re-run LLM, measure output divergence) implemented in `vectorlens/attribution/perturbation.py`. This is highly accurate but expensive: N chunks = N+1 API calls.

Currently, perturbation is **never triggered** — it exists in the codebase but is completely disconnected from the UI. Users don't know it exists and can't opt-in.

This issue is to expose perturbation via a **"Run Deep Attribution" button** in the dashboard:
1. User views a session
2. Clicks "Run Deep Attribution" button
3. Progress bar appears: "Analyzing 5 of 10 chunks..."
4. Dashboard updates with detailed perturbation scores (per chunk)
5. Replaces simple similarity scores with true causal attribution

**Why this matters**: For critical hallucinations, perturbation attribution is gold — it shows *exactly* which chunks cause output changes. Cost is acceptable for one-off debugging.

**Reference files**:
- Perturbation logic: `vectorlens/attribution/perturbation.py` — `PerturbationAttributor` class
- Current attribution UI: `dashboard/src/components/` (e.g., chunks panel)
- Session state: `vectorlens/types.py` — `Session` and `AttributionResult` dataclasses
- API endpoints: `vectorlens/server/api.py`

### Acceptance Criteria
- [ ] New API endpoint: `POST /api/sessions/{id}/deep-attribution` triggers perturbation async job
- [ ] Returns job ID immediately (non-blocking)
- [ ] Endpoint `GET /api/jobs/{id}` returns progress: `{"status": "running", "progress": "5/10", "eta_seconds": 30}`
- [ ] When complete: updates session's `AttributionResult` with perturbation scores
- [ ] Dashboard button: "Run Deep Attribution" in session header (disabled if already running)
- [ ] Shows progress bar with ETA: "Analyzing 5 of 10 chunks (~2 min remaining)"
- [ ] On completion: updates chunk panel to show perturbation scores instead of similarity
- [ ] Cancel button: allows user to stop long-running job

### Implementation Notes
- Perturbation is async (CPU-bound mostly, network for LLM calls) — use `asyncio.Task` to run in background
- Store job state in session: `job_id`, `status`, `progress`, `results`
- Estimate LLM call time per chunk: track actual latency, display ETA
- Error handling: if one perturbation call fails, skip that chunk and continue
- UX: disable button while running, show cancel option
- Tests: mock LLM calls, verify job state transitions correctly

---

## Issue 7: Cross-Session Comparison

**Labels:** `enhancement`, `dashboard`, `ui`
**Milestone:** v0.3.0

### Description

When iterating on RAG systems, developers run the same query multiple times with different prompts, retrievers, or chunk sizes. Manually comparing sessions is tedious.

Add a **cross-session comparison view**: select 2+ sessions → see side-by-side diff of:
- Groundedness scores (% grounded in chunks)
- Hallucination flags (which sentences differed)
- Chunk attribution rankings (does the same chunk rank high in both?)
- Token counts and latency
- Output text diff (words that differ between responses)

**Example use case**:
```
Prompt v1: "Answer based only on provided context"
  → Groundedness: 75%, hallucinated: 1 sentence

Prompt v2: "Answer comprehensively and acknowledge gaps"
  → Groundedness: 82%, hallucinated: 0 sentences
  → This version is better!
```

**Why this matters**: RAG is iterative. Developers make small changes (prompt tweaks, chunk selection, retriever tuning) and need to see the impact. Side-by-side comparison makes this fast and visual.

**Reference files**:
- Session list UI: `dashboard/src/components/SessionList.tsx` (or similar)
- Session detail view: `dashboard/src/components/SessionDetail.tsx`
- API: `vectorlens/server/api.py` for fetching multiple sessions at once

### Acceptance Criteria
- [ ] Dashboard: "Compare Sessions" button (visible in session list or history)
- [ ] Multi-select UI: checkbox to select 2+ sessions
- [ ] Comparison view shows table: columns for each session, rows for metrics (groundedness, hallucinations, tokens, latency, cost)
- [ ] Text diff: side-by-side output comparison (highlight changed words)
- [ ] Chunk attribution table: show top-5 chunks from each session, highlight ranking differences
- [ ] Color coding: green for better, red for worse, neutral if equal
- [ ] Copy comparison to clipboard (for sharing in Slack/issue)
- [ ] Filter: show only sessions with same query (optional, nice-to-have)

### Implementation Notes
- Frontend: React component for multi-select, diff rendering
- Diff library: consider `diff-match-patch` for word-level diffs
- API endpoint: `GET /api/sessions/compare?ids=id1,id2,id3` returns comparison data
- Comparison payload: include all needed metrics to avoid multiple API calls
- Performance: if comparing many sessions, load metrics on demand (not all upfront)
- UX: default to most recent 2 sessions for quick comparison

---

## Issue 8: SQLite Session Persistence

**Labels:** `enhancement`, `help wanted`, `backend`
**Milestone:** v0.3.0

### Description

Currently, all sessions are stored **in-memory only**. When the server restarts, all debugging data is lost. This is by design for a local dev tool, but it's inconvenient:
- Server crashes (bug, power loss) → lost investigation
- Want to review old sessions weeks later → can't (localStorage cleared)
- Collaborators can't access shared sessions

This issue is to add **optional SQLite persistence**:

```python
import vectorlens

# With persistence
url = vectorlens.serve(persist="./sessions.db")

# Without (current behavior)
url = vectorlens.serve()  # In-memory only (default)
```

When enabled, all sessions are durably stored. Survives server restarts and browser clears.

**Why this matters**: Builds trust and usability. Users can leave VectorLens running and reliably access old sessions. Makes it feasible to use for long-term RAG monitoring.

**Reference files**:
- Session structure: `vectorlens/types.py` — `Session`, `LLMRequestEvent`, etc.
- In-memory store: `vectorlens/session_bus.py` — `SessionBus` class
- API endpoints: `vectorlens/server/api.py` — list, get, delete operations

### Acceptance Criteria
- [ ] New module: `vectorlens/persistence/sqlite_backend.py`
- [ ] `SQLiteBackend` class with methods: `save_session()`, `load_session()`, `list_sessions()`, `delete_session()`
- [ ] Schema: table for sessions, request events, response events, attribution results
- [ ] Update `SessionBus` to delegate persistence to backend (if provided)
- [ ] Add `persist: str | None = None` parameter to `vectorlens.serve()`
- [ ] Auto-load all sessions on startup if persistence enabled
- [ ] Dashboard shows all persisted sessions (not just in-memory ones)
- [ ] Cleanup: old sessions (>30 days) can be pruned (manual command or config)
- [ ] Tests: verify sessions persist across restart simulation

### Implementation Notes
- Use `aiosqlite` (async SQLite) for non-blocking database I/O
- Schema versioning: future-proof for schema changes
- Migration: if user enables persistence after collecting in-memory sessions, migrate them
- Concurrency: multiple VectorLens instances accessing same DB should work (use sqlite journal)
- Compression: optionally compress full response text in DB to save space
- Cleanup: add CLI command `vectorlens cleanup --older-than=30d` to remove old sessions
- Tests: use in-memory SQLite (`:memory:`) for test isolation

---

## Issue 13: Real-Time Streaming Dashboard

**Labels:** `enhancement`, `dashboard`, `v0.2.0`
**Milestone:** v0.2.0

### Description

Currently, the dashboard updates on completion: user queries RAG, LLM generates response, attribution finishes, then dashboard shows results. This is a 5–15 second lag in the developer's mental model.

For **streaming responses** (where tokens arrive incrementally), VectorLens should update the output panel in real-time via WebSocket. As the LLM streams tokens, the dashboard should show "Analyzing..." placeholder for each sentence, immediately replaced by red/green hallucination highlighting when that sentence's attribution is computed.

This closes the feedback loop: developer sees results arriving live, not waiting for the full response + analysis.

**Why this matters**: Fast feedback = faster iteration. Streaming is the default for modern RAG UIs. VectorLens must feel responsive and real-time, not slow and batch-like.

**Reference files**:
- WebSocket connection: `dashboard/src/hooks/useWebSocket.ts`
- Output panel: `dashboard/src/components/OutputPanel.tsx`
- API events: `vectorlens/server/api.py::websocket_endpoint()`
- Pipeline attribution: `vectorlens/pipeline.py::_on_attribution_complete()`

### Acceptance Criteria

- [ ] When streaming response detected, emit partial `LLMResponseEvent` to WebSocket as tokens arrive
- [ ] Dashboard shows partial output with "Analyzing..." gray placeholder for each sentence
- [ ] As attribution completes for each sentence, update that sentence's styling (red/green)
- [ ] Final response text and full attribution included in final `LLMResponseEvent`
- [ ] Dashboard smoothly transitions from placeholder to styled output (no flicker)
- [ ] Works for both single-shot and streaming responses (backward compatible)
- [ ] Tests verify WebSocket partial messages are sent during streaming

### Implementation Notes

- Streaming responses come as `LLMResponseEvent` with `output_text` initially partial, updated over time
- Use `streaming: bool` flag in `LLMResponseEvent` to signal real-time updates
- Dashboard should batch WebSocket updates (e.g., 100ms debounce) to avoid overwhelming the UI
- Graceful fallback: if streaming fails, show full response as normal (already handled by httpx transport)
- Test with OpenAI `stream=True` and ensure partial events arrive before final event

---

## Issue 14: `vectorlens.assert_grounded()` Testing Helper

**Labels:** `good first issue`, `testing`, `v0.2.0`
**Milestone:** v0.2.0

### Description

VectorLens detects hallucinations but doesn't provide a **testing assertion**. Developers want to inline RAG quality checks:

```python
from vectorlens.testing import assert_grounded

def test_rag_answers_correctly():
    response = rag.query("What is attention?")
    chunks = rag.get_chunks()

    # Fails if groundedness < 80%
    assert_grounded(response, chunks, min_score=0.8, message="RAG should be grounded")
```

This should work in:
- pytest test functions
- Jupyter notebooks
- Standalone scripts

**Why this matters**: Makes it trivial to add RAG quality gates to test suites. Developers can fail builds on hallucination regressions, not just functional bugs.

**Reference files**:
- Testing module (new): `vectorlens/testing.py`
- Detection logic: `vectorlens/detection/hallucination.py::HallucinationDetector.detect()`
- Attribution logic: `vectorlens/types.py` — `OutputToken`, `AttributionResult`

### Acceptance Criteria

- [ ] New module: `vectorlens/testing.py` with `assert_grounded()` function
- [ ] Signature: `assert_grounded(response: str, chunks: list[str] | list[RetrievedChunk], min_score: float = 0.8, message: str = "Response not grounded")`
- [ ] Detects hallucinations using `HallucinationDetector`
- [ ] Calculates overall groundedness as `1.0 - (hallucinated_count / total_sentences)`
- [ ] Raises `AssertionError` with descriptive message if `groundedness < min_score`
- [ ] Message includes: actual groundedness, threshold, which sentences were hallucinated
- [ ] Works in pytest (normal function), Jupyter (prints results), standalone scripts
- [ ] Tests verify assertion passes/fails correctly

### Implementation Notes

- Use `HallucinationDetector()` directly (no need to go through server)
- `chunks` parameter can be raw strings or `RetrievedChunk` objects — convert to `RetrievedChunk` with empty metadata if needed
- Message format: `f"AssertionError: Response not grounded (actual: 65%, required: 80%)\nHallucinated: ['sentence 1', 'sentence 2']"`
- No dependency on `vectorlens.serve()` — works standalone
- Lazy-load the sentence-transformers model on first call (same as detection pipeline)

---

## Issue 15: Prompt A/B Comparison UI

**Labels:** `enhancement`, `dashboard`, `v0.2.0`
**Milestone:** v0.2.0

### Description

RAG developers iterate on prompt wording: "Should I say 'Answer based on context only' or 'Use only the provided documents'?" Both sound plausible, but only one might reduce hallucinations.

Add a **comparison view**: Select two sessions in the sidebar → side-by-side panel showing:
- Prompt text (left vs right)
- Output diff (highlight differences)
- Groundedness delta: `+12% groundedness`, `-2 hallucinated sentences`
- Token counts and latency comparison
- Top-5 chunk attributions side-by-side

**Why this matters**: A/B testing prompts is iterative RAG development. Visual comparison makes it obvious which version performed better.

**Reference files**:
- Session sidebar: `dashboard/src/components/SessionList.tsx`
- Session detail: `dashboard/src/components/SessionDetail.tsx`
- API: `vectorlens/server/api.py` — add `/api/sessions/compare` endpoint
- Types: `vectorlens/types.py` — `Session`, `AttributionResult`

### Acceptance Criteria

- [ ] Dashboard UI: "Compare" button visible when 2+ sessions selected (multi-select checkboxes)
- [ ] Comparison view shows table with columns: Metric | Session A | Session B | Delta
- [ ] Rows: Groundedness%, Hallucinated Count, Output Length, Tokens, Latency (ms), Cost (USD)
- [ ] Color coding: green if better in B, red if worse, gray if equal
- [ ] Text diff: side-by-side output comparison with changed words highlighted
- [ ] Chunk attribution: top-5 chunks from each session, sorted by score, highlight ranking differences
- [ ] "Copy comparison" button copies table as markdown (for sharing in Slack/issue)
- [ ] Filter (optional): show only sessions with same query text

### Implementation Notes

- Frontend: React component `ComparisonView.tsx` with multi-select logic
- Diff library: use `diff-match-patch` for word-level diffs
- API endpoint: `GET /api/sessions/compare?ids=id1,id2` returns comparison payload
- Comparison logic: compute metrics from both sessions, compute deltas
- Performance: load metrics on-demand (not all sessions at once)
- UX: pre-select most recent 2 sessions for quick comparison

---

## Issue 16: Human Annotation Layer for Ground Truth Labeling

**Labels:** `enhancement`, `dashboard`, `v0.2.0`
**Milestone:** v0.2.0

### Description

VectorLens detects hallucinations automatically, but it sometimes gets it wrong. A sentence might look hallucinated (low semantic match) but actually be correct. Conversely, a grounded sentence might contain a subtle factual error VectorLens misses.

Add a **right-click annotation UI** for developers to provide ground truth:

```
Right-click on hallucinated sentence (red background)
  → "Actually correct" / "Confirmed hallucination" / "Unsure"

Annotations persist in localStorage + can be exported as labeled dataset
```

This lets developers:
1. Correct VectorLens's mistakes (improve future versions)
2. Build a labeled dataset of "good" vs "bad" sentences for fine-tuning
3. Track confidence in the automated detection

**Why this matters**: Ground truth labels = ability to validate and improve hallucination detection. Also generates training data for custom detectors.

**Reference files**:
- Output panel: `dashboard/src/components/OutputPanel.tsx`
- Session state: `vectorlens/types.py` — add `annotations: dict[str, str]` to `Session`
- Storage: `dashboard/src/lib/storage.ts` — localStorage persistence

### Acceptance Criteria

- [ ] Right-click on any output sentence → context menu with three options: "Correct", "Hallucinated", "Unsure"
- [ ] Selected annotation persists in session state (localStorage if no persistence, SQLite if persistence enabled)
- [ ] Annotated sentences show badge (e.g., green checkmark for "Correct", red X for "Hallucinated")
- [ ] UI shows annotation count: "3 annotated, 2 correct, 1 hallucinated"
- [ ] Export button: `GET /api/sessions/{id}/annotations` returns JSON list of `{sentence, is_hallucinated_by_model, human_label, confidence}`
- [ ] Annotations can override automatic detection in dashboard display (optional: show both)

### Implementation Notes

- Context menu: use `contextmenu` event, position menu near cursor
- Annotation storage: `annotations: {sentence_text: label}` in Session object
- Badge styling: green (Correct), red (Hallucinated), yellow (Unsure)
- Export format: CSV or JSON, suitable for training datasets
- Future use: compare model vs human labels to measure detection accuracy

---

## Issue 17: pytest Plugin for RAG Test Assertions

**Labels:** `enhancement`, `testing`, `help wanted`, `v0.3.0`
**Milestone:** v0.3.0

### Description

VectorLens can power **regression testing for RAG pipelines**. Imagine:

```python
# conftest.py
import pytest
import vectorlens

@pytest.fixture(scope="session", autouse=True)
def setup_vectorlens():
    vectorlens.serve()
    yield
    vectorlens.stop()

# test_rag.py
@pytest.mark.rag(groundedness_threshold=0.8)
def test_response_is_grounded():
    response = rag.query("What is attention?")
    # Auto-asserts: response must be 80%+ grounded
    # Fails test if not
```

This issue is to create a **pytest plugin** (`pytest-vectorlens`) that:
1. Auto-starts VectorLens on test session init
2. Provides `@pytest.mark.rag` decorator with auto-assertions
3. Generates test reports with hallucination metrics
4. Integrates with pytest reporting (summary shows groundedness scores)

**Why this matters**: RAG quality is not just "did the code run" — it's "are the answers grounded?" Tests should validate quality, not just functionality.

**Reference files**:
- Session API: `vectorlens/server/api.py` — get session metrics
- Testing helpers (to be built): `vectorlens/testing.py`
- pytest hook pattern: `pytest_configure()`, `pytest_sessionfinish()`, custom markers

### Acceptance Criteria

- [ ] New package: `pytest-vectorlens` (separate from main, but co-published)
- [ ] Entry point: `pytest11 = {"vectorlens": "pytest_vectorlens.plugin"}`
- [ ] Hook `pytest_configure()`: auto-start `vectorlens.serve()`
- [ ] Hook `pytest_sessionfinish()`: stop VectorLens cleanly
- [ ] Decorator: `@pytest.mark.rag(groundedness_threshold=0.8)` on test functions
- [ ] Decorator triggers auto-assertion on test completion: collect all sessions from test, check groundedness >= threshold
- [ ] Test report: show summary line: "RAG Quality: 4 tests passed, 2 failed (avg groundedness: 82%)"
- [ ] CLI option: `pytest --vectorlens-threshold=0.8` to set global threshold
- [ ] CLI option: `pytest --vectorlens-disable` to skip VectorLens
- [ ] Example test file included in docs

### Implementation Notes

- pytest hooks: use `pytest_configure()` and `pytest_sessionfinish()` hooks
- Session tracking: after each test, query VectorLens API for sessions created during that test
- Assertion logic: use `assert_grounded()` helper from `vectorlens/testing.py`
- Report integration: add custom pytest plugin section via `pytest_terminal_summary()` hook
- Headless CI: auto-disable browser open with `open_browser=False`
- Documentation: add to README with GitHub Actions example

---

## Issue 18: GitHub Actions Groundedness Regression Check

**Labels:** `enhancement`, `integration`, `help wanted`, `v0.3.0`
**Milestone:** v0.3.0

### Description

When a PR changes RAG code (prompt, retrieval strategy, chunk size), groundedness can regress. A PR that "adds a feature" might accidentally reduce answer quality by 15%.

Add a **GitHub Actions workflow** that:
1. Runs RAG test suite (via pytest-vectorlens)
2. Compares groundedness to baseline (`vectorlens.baseline.json` in repo)
3. Comments on PR with results: `Groundedness: 85% (was 92%, -7% regression) ⚠️`
4. Fails PR if regression > threshold (default 5%)

**Why this matters**: Quality-aware CI/CD. No more shipping hallucination regressions.

**Reference files**:
- Baseline file format (new): `.github/vectorlens.baseline.json` (example)
- Workflow template (new): `.github/workflows/vectorlens.yml`
- pytest plugin: `pytest-vectorlens` (from issue #17)
- API: ability to export session metrics as JSON

### Acceptance Criteria

- [ ] Workflow template: `.github/workflows/vectorlens-check.yml` (can be copy-pasted into repos)
- [ ] Workflow runs pytest-vectorlens on PR branches
- [ ] Compares metrics (groundedness%, hallucination_count) to baseline
- [ ] Computes delta: `(current - baseline) / baseline * 100`
- [ ] Posts GitHub comment on PR: `Groundedness: 85% (was 92%, -7% regression) ⚠️ FAILED`
- [ ] Fails PR if `|delta| > regression_threshold` (default 5%)
- [ ] Allows overriding threshold via workflow input: `max-regression: 10`
- [ ] Comments include: metric breakdown (tokens, latency, cost)
- [ ] Baseline file format: simple JSON `{"groundedness": 0.92, "hallucination_count": 2, "avg_latency_ms": 1200}`
- [ ] Workflow also updates baseline on `main` branch (after PR merge)

### Implementation Notes

- Workflow should install VectorLens: `pip install vectorlens[pytest]`
- Run tests: `pytest --vectorlens-report=metrics.json tests/test_rag.py`
- Compare metrics: Python script to read baseline and current metrics, compute delta
- GitHub comment: use GitHub Actions API (via `gh` CLI or actions)
- Baseline tracking: commit `vectorlens.baseline.json` to repo
- Documentation: add example to README with copy-paste workflow

---

## Issue 19: SQLite Session Persistence

**Labels:** `enhancement`, `help wanted`, `v0.3.0`
**Milestone:** v0.3.0

### Description

Currently, all sessions live **in-memory only**. Server restart = lost data. This is inconvenient:

- Debugging a complex hallucination, server crashes, investigation lost
- Want to review sessions from yesterday → impossible (localStorage cleared)
- Multiple developers need to share debugging sessions → can't (in-memory only)

Add **optional SQLite persistence**:

```python
import vectorlens

# With persistence (new!)
url = vectorlens.serve(persist="./sessions.db")

# Without (current behavior)
url = vectorlens.serve()  # In-memory only (default)
```

Sessions now survive server restarts and browser clears.

**Why this matters**: Builds trust in the tool. Developers can leave VectorLens running long-term, reliably access old sessions. Makes it feasible as an always-on RAG monitoring solution.

**Reference files**:
- Session structure: `vectorlens/types.py` — `Session`, `LLMRequestEvent`, `AttributionResult`
- In-memory backend: `vectorlens/session_bus.py` — `SessionBus._sessions` dict
- API endpoints: `vectorlens/server/api.py` — `list_sessions()`, `get_session()`, `delete_session()`
- Server init: `vectorlens/__init__.py::serve()` function

### Acceptance Criteria

- [ ] New module: `vectorlens/persistence/sqlite_backend.py` with `SQLiteBackend` class
- [ ] Methods: `save_session()`, `load_session()`, `list_sessions()`, `delete_session()`
- [ ] Database schema: tables for `sessions`, `llm_requests`, `llm_responses`, `attributions`
- [ ] Update `SessionBus.__init__()` to accept optional `backend` parameter
- [ ] Add `persist: str | None = None` parameter to `vectorlens.serve()`
- [ ] On startup with `persist=path`: auto-load all sessions from DB
- [ ] Dashboard shows all persisted sessions (with "Cached" badge for old ones)
- [ ] Sessions auto-sync to DB (not just in-memory)
- [ ] Cleanup: manual command `vectorlens cleanup --older-than=30d` removes old sessions
- [ ] Tests: verify sessions persist across server restart simulation

### Implementation Notes

- Use `aiosqlite` for async SQLite (non-blocking I/O)
- Schema versioning: plan for future schema changes
- Migration: if user enables persistence after collecting in-memory sessions, migrate them on first run
- Concurrency: multiple VectorLens instances can safely read same DB (SQLite handles locking)
- Compression: optionally gzip full response text in DB to save space
- Cleanup: implement CLI command and automatic pruning (optional)
- Tests: use in-memory SQLite (`:memory:`) for isolation

---

## Issue 20: Self-Contained Session Export (`vectorlens share`)

**Labels:** `enhancement`, `good first issue`, `v0.3.0`
**Milestone:** v0.3.0

### Description

To share a hallucination report with teammates, users currently:
1. Screenshot the dashboard
2. Manually copy-paste text
3. Send via Slack/email

This is tedious and error-prone. Add a **CLI command** that generates a self-contained HTML file:

```bash
vectorlens share abc123-def456
# Outputs: session_abc123-def456.html (2–3 MB)

# Share as email attachment or Slack upload
# Recipients open HTML file in browser → see full attribution dashboard
# No server needed to view
```

**Why this matters**: Async collaboration without tab-switching or screenshots. Teammates can explore hallucinations independently.

**Reference files**:
- Dashboard static assets: `dashboard/dist/`
- Session API: `vectorlens/server/api.py::get_session()`
- CLI entry point: `vectorlens/__main__.py` or `setup.py` console_scripts

### Acceptance Criteria

- [ ] New CLI command: `vectorlens share <session_id>` (entry point in `setup.py`)
- [ ] Generates single HTML file with embedded React bundle + session data
- [ ] File naming: `session_{session_id}.html` (no spaces, safe for email)
- [ ] File size: keep under 5MB (compress assets if needed)
- [ ] Opens in any browser without server (standalone file)
- [ ] Shows full attribution UI: chunks panel, groundedness meter, output with highlighting
- [ ] Includes metadata: timestamp, threshold used, vectorlens version
- [ ] Works with both in-memory and persisted sessions
- [ ] Error handling: if session not found, print helpful message

### Implementation Notes

- Build approach: embed dashboard/dist files + session JSON into HTML template via `<script>window.__SESSION_DATA__ = {...}</script>`
- Compression: use gzip for JavaScript bundle to reduce file size
- Styling: ensure all CSS is inlined (Tailwind classes already built into dist/main.css)
- React hydration: dashboard loads session from `window.__SESSION_DATA__` instead of API
- File size: target <3MB (test with real session data)
- CLI: add to `vectorlens/__main__.py` with argparse
- Documentation: show example in README: "Share debugging sessions without screenshots"

---

## Issue 9: True Token-Level Attribution via Attention

**Labels:** `research`, `hard`, `detection`
**Milestone:** v0.4.0

### Description

Current hallucination detection is **sentence-level**: if a sentence doesn't match retrieved chunks, it's flagged as hallucinated. But a sentence can be *mostly* correct with one hallucinated word mixed in:

```
Output: "Transformers use self-attention to compute token relationships across planets."
         (^ "planets" is hallucinated, but whole sentence flagged as grounded because 0.85 similarity to chunk)
```

True **token-level attribution** would identify the exact word that's hallucinated. This requires extracting **attention weights** from the LLM's internal computation.

VectorLens already has skeleton code (`vectorlens/attention.py`) for computing attention rollout on local HuggingFace models, but it's **completely disconnected** from the main pipeline.

This issue is to wire it up:
1. After LLM response is generated, check if model weights are available locally
2. If yes (e.g., using HuggingFace transformers or Ollama with local model), extract attention
3. Compute per-token attribution via attention rollout
4. Dashboard shows **token-level heatmap**: each word colored by its hallucination likelihood
5. Fall back to sentence-level for proprietary models (OpenAI, Anthropic)

**Why this matters**: Token-level attribution is the holy grail of RAG debugging. Users see *exactly* which words are hallucinated, not just sentences.

**Reference files**:
- Skeleton code: `vectorlens/attention.py` (currently unused)
- HuggingFace integration: `vectorlens/interceptors/transformers_patch.py`
- Detection pipeline: `vectorlens/pipeline.py` — where attribution runs

### Acceptance Criteria
- [ ] Implement `AttentionAttributor` class in new module or extend `attention.py`
- [ ] Method: `extract_attention_weights(model, input_ids, output_ids) -> Tensor`
- [ ] Method: `compute_token_attribution(attention, input_ids, output_ids) -> list[dict]` with scores per token
- [ ] Wire into pipeline: after LLM response, try to get model weights
- [ ] For local HuggingFace models: use attention rollout (existing algorithm in `attention.py`)
- [ ] For proprietary models (OpenAI, Anthropic): fall back to sentence-level (current behavior)
- [ ] Dashboard: token-level heatmap visualization (color gradient per word)
- [ ] Tests: verify attention extraction works with real HuggingFace models (integration test)

### Implementation Notes
- Attention rollout algorithm: average attention across heads and layers
- Input IDs: need to trace which generated tokens correspond to which input chunks
- Edge case: recurrent models (like RNN) have different attention structure — handle gracefully
- Performance: attention extraction is CPU-bound but single-threaded; shouldn't block main code
- Privacy: never send attention weights to dashboard (keep local only)
- Start with decoder-only models (GPT-2, Mistral) — easier than encoder-decoder (T5)

---

## Issue 10: MCP Server Integration

**Labels:** `enhancement`, `integration`, `research`
**Milestone:** v0.4.0

### Description

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is a new standard for AI assistants (Claude, Cursor, etc.) to call external tools. VectorLens should expose itself as an **MCP server** so Claude Code / Cursor can call:

```
claude> @vectorlens What hallucinated in session abc123?
claude> @vectorlens Compare sessions xyz and abc
claude> @vectorlens Export session abc as JSON
```

This embeds RAG debugging directly into the IDE / chat interface.

**Why this matters**: Developers spend time in Claude Code and Cursor. Exposing VectorLens via MCP means they can debug RAG without tab-switching. Seamless integration with their development workflow.

**Reference files**:
- MCP spec: https://modelcontextprotocol.io/
- Python SDK: `fastmcp` library (or native MCP server implementation)
- Session API: `vectorlens/server/api.py` — already has all the data

### Acceptance Criteria
- [ ] New module: `vectorlens/mcp_server.py`
- [ ] Implements MCP server spec (resources + tools)
- [ ] Resources: `session://abc123` (access session data)
- [ ] Tools: `query_session(session_id, query_str)` — semantic search over hallucinations
- [ ] Tools: `export_session(session_id)` — return JSON
- [ ] Tools: `compare_sessions(ids)` — return comparison data
- [ ] Starts alongside FastAPI server (separate port or Unix socket)
- [ ] Supports both `fastmcp` and native MCP SDK (choose simplest)
- [ ] Tests: mock MCP client, verify requests/responses

### Implementation Notes
- MCP server can be sync or async — use async for compatibility with FastAPI event loop
- Resource representation: URI-style (`session://id/details`, `session://id/chunks`)
- Tools should be simple and focused (not overload with options)
- Error handling: malformed query → graceful error message (not exception)
- Documentation: add instructions to README for connecting Claude Code
- Start simple: just expose data retrieval, not triggering new analyses (that comes later)

---

## Issue 11: Automatic Prompt Improvement Suggestions

**Labels:** `research`, `enhancement`, `advanced`
**Milestone:** v0.4.0

### Description

When VectorLens detects hallucinations, it *identifies the problem* but doesn't suggest *solutions*. Developers are left to manually brainstorm: "How do I fix this?"

This issue is to use an LLM to **suggest concrete improvements**:

1. **Better retrieval queries**: If hallucination is caused by bad retrieval, suggest more specific queries
   ```
   Current: "What is X?"
   Better: "What is X in the context of Y?" (more specific)
   ```

2. **Prompt changes**: Rewrite the system prompt to reduce hallucinations
   ```
   Current: "Answer the user's question."
   Better: "Answer using only the provided context. Say 'I don't know' if not found."
   ```

3. **Knowledge base gaps**: Identify which chunks are missing
   ```
   Hallucinated: "The system uses GPU acceleration"
   Missing chunk: "Performance optimization section from docs"
   Suggestion: Add GPU acceleration docs to knowledge base
   ```

**Why this matters**: Suggestion = 10× faster iteration. Instead of "I see hallucinations, now what?", developers get "Try adding this phrase to your prompt" or "Retrieve chunks about performance optimization."

**Reference files**:
- Hallucination detection: `vectorlens/detection/hallucination.py`
- Session data: `vectorlens/types.py` — has all context needed
- API: `vectorlens/server/api.py` — expose suggestions

### Acceptance Criteria
- [ ] New module: `vectorlens/suggestions/improvements.py`
- [ ] Function: `suggest_retrieval_queries(hallucinated_text, context) -> list[str]`
- [ ] Function: `suggest_prompt_changes(hallucinated_text, original_prompt) -> list[str]`
- [ ] Function: `suggest_knowledge_base_gaps(hallucinated_text, available_chunks) -> list[str]`
- [ ] API endpoint: `POST /api/sessions/{id}/suggestions` returns all suggestions
- [ ] Uses an LLM (configurable: default to Claude or GPT-4o)
- [ ] Dashboard: "Get Suggestions" button next to hallucinated sentences
- [ ] Shows suggestions in readable format (bullet points, with explanations)
- [ ] Optional: allow user to vote on suggestions (feedback for future improvements)

### Implementation Notes
- Use a strong LLM: Claude 3.5 Sonnet or GPT-4 (for quality)
- Prompt engineering: craft detailed system prompt to generate actionable suggestions (not generic advice)
- Cost: each suggestion requires 1 LLM call (~$0.01) — disable by default, opt-in only
- Context window: include hallucinated sentence, surrounding sentences, top-5 chunks for context
- Async: suggestions are computed on-demand, not auto-triggered (expensive)
- Validation: filter low-confidence suggestions before showing to user

---

## Issue 12: CI/CD Integration — pytest Plugin

**Labels:** `enhancement`, `testing`, `integration`
**Milestone:** v0.3.0

### Description

VectorLens is a runtime debugging tool, but it could also be a **test assertion tool**. Imagine:

```python
# conftest.py
import pytest
import vectorlens

@pytest.fixture(scope="session", autouse=True)
def enable_vectorlens():
    vectorlens.serve()
    yield
    vectorlens.stop()

# test_rag.py
from vectorlens.testing import assert_groundedness

def test_rag_response_is_grounded():
    response = rag.query("What is attention?")
    chunks = rag.get_chunks()

    # Fails if groundedness < 80%
    assert_groundedness(response, chunks, min_score=0.8)
```

This issue is to create a **`pytest-vectorlens` plugin** that:
1. Auto-starts VectorLens during test runs (no manual setup)
2. Captures all RAG calls made by tests
3. Provides assertion functions: `assert_groundedness()`, `assert_no_hallucinations()`
4. Fails tests if RAG quality drops below thresholds (regression testing)
5. Generates test reports showing hallucinations found

**Why this matters**: Tests should validate quality, not just functionality. RAG regression tests catch when new chunks break existing answers. VectorLens is the missing piece of RAG test coverage.

**Reference files**:
- Session API: `vectorlens/types.py` and `vectorlens/server/api.py`
- Detection: `vectorlens/detection/hallucination.py`
- Pytest plugin pattern: see `pytest` plugin architecture (hooks in `conftest.py`)

### Acceptance Criteria
- [ ] New package: `pytest-vectorlens` (separate from main package, but published together)
- [ ] Module: `pytest_vectorlens/plugin.py` with pytest hooks
- [ ] Hook: `pytest_configure()` to auto-start VectorLens
- [ ] Hook: `pytest_sessionfinish()` to stop VectorLens
- [ ] Module: `pytest_vectorlens/assertions.py` with assertion functions
  - [ ] `assert_groundedness(response, chunks, min_score=0.8) -> bool`
  - [ ] `assert_no_hallucinations(response, chunks) -> bool`
  - [ ] `assert_attribution(response, chunks, expected_chunk_ids=[...]) -> bool`
- [ ] Test report: add to `pytest` summary showing hallucinations found in tests
- [ ] Integration: `pytest --vectorlens-threshold=0.8` CLI option to set threshold
- [ ] Tests: example test file showing pytest-vectorlens in action
- [ ] Documentation: add to README and CONTRIBUTING guide

### Implementation Notes
- Pytest plugin entry point: add `entry_points` in `setup.py` or `pyproject.toml`
- Auto-start: use `pytest_configure()` hook, store session in plugin state
- Assertion functions: accept response text and chunks, use `HallucinationDetector` directly
- Report integration: use `pytest` hooks to add custom section to terminal output
- CLI args: add options like `--vectorlens-threshold`, `--vectorlens-disable`
- CI/CD: document how to use in GitHub Actions / GitLab CI (add examples)
- For headless CI: disable browser auto-open, serve on localhost only

---
