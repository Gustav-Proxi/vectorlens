# VectorLens Vision

## The Vision Statement

VectorLens is the RAG debugger that should have existed from the start. The goal: make hallucination debugging as fast and obvious as print debugging — no cloud account, no SDK rewrites, no 45-minute setup.

By v1.0, VectorLens should be the default debugging layer for any Python developer building RAG — the way pytest is the default test runner. You install it once without thinking about it, and it's just there when you need it.

---

## The Problem We're Solving

Every developer building RAG hits this wall:

**Scenario**: Your chatbot just told a user something completely false. It sounds confident. It sounds grounded. But it's a hallucination.

Your instinct is to debug it. But debugging RAG is hell:

- **Where did it come from?** You don't know which chunk caused it. Was it retrieval? Was it the LLM inventing something? Both?
- **Add print statements.** You get 50 lines of output. The chunk you retrieved looks fine. The LLM response looks plausible. You still don't know.
- **Sign up for LangSmith.** 20 minutes of setup. API keys everywhere. Cloud dashboard. Your prompts and chunks leave your machine for Observability SaaS infrastructure. Compliance meeting incoming.
- **Try Arize Phoenix.** More instrumentation. OTEL setup. Traces sent somewhere. More cloud dependency.
- **Give up and guess.** You tweak the prompt. You change the retrieval model. You hope it helps. You deploy. It doesn't.

VectorLens solves this in **two lines of code**:

```python
import vectorlens
vectorlens.serve()
```

Then you ask your RAG question normally. VectorLens watches what happens. It detects hallucinations. It shows you exactly which chunk caused which hallucinated sentence, with attribution scores. Local. Instant. Your data never leaves your machine.

No cloud. No instrumentation. No 45-minute setup.

---

## Design Principles (Non-Negotiables)

### 1. Local-First, Always

User data (prompts, responses, chunks, conversations) never leaves the machine. Ever. No cloud option, not even optional. VectorLens is a development tool for *your* machine, not a SaaS.

**Why**: Privacy, speed, and developer freedom. In v0.1.3, a developer can debug their RAG with proprietary prompts and proprietary data without a compliance meeting. That matters.

**What this rules out**: Hosted dashboards, cloud-backed sessions, third-party analytics, telemetry. We don't ask where you're debugging. We don't know.

---

### 2. Zero Instrumentation

`import vectorlens; vectorlens.serve()` and your existing code works unchanged. If you need to modify your RAG pipeline to use VectorLens, we've failed.

**How it works**: Monkey-patching at the transport layer (httpx for LLM SDKs, SQLAlchemy for pgvector, etc.). Your code calls OpenAI normally; we intercept at the network layer. You don't need wrapper functions, custom prompts, or special classes.

**What this rules out**: Requiring SDK changes, asking developers to pass a `vectorlens_session` object through their chain, instrumenting every retrieval call manually. If it requires code changes, it's not zero-instrumentation.

---

### 3. Non-Intrusive

VectorLens bugs never crash user code. All instrumentation is wrapped in try/except. Attribution is best-effort, never blocking.

**Why**: A developer's RAG pipeline is production code. VectorLens is a debugger. The debugger should never break the thing it's trying to debug.

**What this means in practice**:
- If hallucination detection fails, we log it silently and move on. The LLM response still arrives.
- If the attribution pipeline gets backed up (queue full), we drop new tasks. We don't block the LLM client.
- If a vector DB interceptor throws, we catch it, log, and fall through to the original call.

---

### 4. Focused

We do one thing: explain why RAG hallucinates. We show which chunks caused which hallucinated sentences, with confidence scores.

We do *not* build:
- Evaluation frameworks (that's promptfoo)
- Red-teaming tools (that's also promptfoo, Agentic, etc.)
- Training data curation platforms (that's Argilla, Label Studio)
- LLMOps dashboards (that's LangSmith, Weights & Biases, MLflow)
- Prompt management systems (that's prompt-versioning tools)

Staying focused means staying excellent. Each of those is a full product. We're not that. We're a debugging tool.

---

### 5. Contributor-Friendly

Every new integration (new LLM provider, new vector DB) should take an afternoon to add, following a clear template. The codebase should be readable by someone who's never seen it before.

**What this means**:
- Clear separation of concerns (interceptors, detection, attribution, server)
- Every interceptor follows the same pattern: inherit `BaseInterceptor`, implement `install()` / `uninstall()`
- Type hints everywhere (not optional)
- Tests for every integration (mocked first, real optional)
- When you read `openai_patch.py`, you understand what to change to add Cohere support

---

## Current State (v0.1.3)

### What's Working

**LLM Provider Coverage**: 9 providers intercepted
- OpenAI (GPT-4o, GPT-4 Turbo, GPT-3.5)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- Google Gemini (SDK v1 & v2)
- Mistral (all models)
- HuggingFace Transformers (local inference)
- LangChain (framework-level: BaseChatModel, BaseRetriever, LCEL)
- Plus any httpx-based SDK (transport-layer coverage)

**Vector DB Coverage**: 6 providers intercepted
- ChromaDB (in-memory and persistent)
- Pinecone (serverless indices)
- FAISS (local dense search)
- Weaviate (enterprise vector DB)
- pgvector (PostgreSQL + SQLAlchemy)
- Custom via manual event API

**Core Features**:
- Streaming support (`stream=True` now captured; SSE chunks intercepted)
- Conversation DAG (`parent_request_id`, `chain_step`, `conversation_id` for multi-turn tracing)
- Attribution pipeline: hallucination detection + conditional LIME-style bounded perturbation
- Smart attribution (skip deep work when fully grounded)
- Robust chunk removal (3-tier fallback for reformatted chunks)
- Real-time dashboard with session history, chunk attribution, groundedness score
- 117+ tests passing

### What's Missing (Honest Gaps)

**Token-level attribution for API models**: OpenAI and Anthropic don't expose attention weights. Current MVP uses sentence-level semantic similarity. True word-level attribution for these models requires either (a) gradient approximation over prompt embeddings (expensive), or (b) waiting for API providers to expose attention. We have a path forward, but it's non-trivial.

**No streaming dashboard**: Tokens appear in the output panel after the full response, not during streaming. User sees "Analyzing..." then full output. We need to emit partial WebSocket events from `_StreamingResponseWrapper` to fix this; requires careful state management on the dashboard.

**No pytest integration**: Can't assert groundedness in test suites. `assert_grounded(response, chunks, min_score=0.8)` doesn't exist yet. This is planned for v0.2 and is high-impact for CI/CD.

**No persistence**: Sessions are in-memory only. When server restarts, debugging history is lost. SQLite backing is planned for v0.3.

**Token counts approximate for streaming**: Streaming responses don't include exact token counts in SSE metadata. VectorLens estimates from word count.

**No team/sharing features yet**: Single-user local debugging. No way to export a session and share with a teammate without screenshot. Planned for v0.3.

---

## The Roadmap (Detailed)

### v0.2.0 — The Developer Loop (Target: 4–6 weeks)

Make the day-to-day debugging loop tight and obvious. The goal: iterate fast on RAG quality changes.

#### 1. Real-Time Streaming Dashboard
**What**: Tokens appear in the output panel as they stream, live via WebSocket.

**Why it matters**: Streaming is standard for production RAG UIs. Currently the output panel stays blank during streaming and fills in after. This breaks the "I can watch it debug in real-time" experience.

**How**: Modify `_StreamingResponseWrapper` to emit partial `LLMResponseEvent` payloads as tokens arrive. Dashboard receives these via WebSocket and appends to output panel. When stream ends, final attribution computes and displays.

**Ownership**: `vectorlens/interceptors/httpx_transport.py` (streaming capture), `vectorlens/server/app.py` (WebSocket broadcasts), `dashboard/src/components/OutputHighlighter.tsx` (streaming UI).

#### 2. Inline Assertions: `assert_grounded()`
**What**: `from vectorlens.testing import assert_grounded; assert_grounded(response, chunks, min_score=0.8)` raises with a readable diff if groundedness < threshold.

**Why it matters**: Developers can add quality gates to their test suites in one line. No server required. Can be used in pytest, notebooks, or standalone scripts.

**How**: Create `vectorlens/testing.py` with `assert_grounded()`. Internally, it runs the detection pipeline synchronously (not in background). Returns a simple bool or raises `AssertionError` with the hallucinated spans listed.

**Ownership**: `vectorlens/testing.py` (new file), tests in `tests/test_testing.py`.

#### 3. Prompt A/B Comparison
**What**: Cmd/Ctrl+click two sessions → side-by-side groundedness diff. `+12% groundedness`, `-2 hallucinated sentences`.

**Why it matters**: Fastest way to know if a prompt change helped. Instead of running a benchmark, you can compare two debug sessions instantly.

**How**: New React component `ComparisonView` showing two sessions side-by-side. Diffs shown for: groundedness %, hallucinated sentence count, chunk attribution differences. Stored in localStorage for easy retrieval.

**Ownership**: `dashboard/src/components/ComparisonView.tsx` (new), sidebar interaction in `SessionList.tsx`.

#### 4. Human Annotation Layer
**What**: Right-click any sentence → mark as "confirm hallucination" / "mark as correct" / "unsure". Annotations persist in localStorage. Export as JSONL fine-tuning dataset.

**Why it matters**: Turns debugging sessions into labeled data. Instead of just debugging, you're crowdsourcing ground truth for future fine-tuning.

**How**: Add right-click context menu to `OutputHighlighter.tsx`. Store annotations in `vectorlens/annotations.json` (localStorage). New API endpoint `GET /api/sessions/{id}/export-annotations` returns JSONL.

**Ownership**: `dashboard/src/components/OutputHighlighter.tsx` (context menu), `vectorlens/server/api.py` (export endpoint).

---

### v0.3.0 — Team & CI (Target: 8–12 weeks)

Scale from solo debugging to team collaboration and automated regression testing.

#### 1. pytest Plugin
**What**: `pip install vectorlens[pytest]` adds a `vectorlens_session` fixture. `@pytest.mark.rag` decorator auto-asserts `session.overall_groundedness >= threshold`.

**Why it matters**: Groundedness becomes a first-class CI metric, like code coverage. Regressions are caught before merging.

**How**: Create `vectorlens/pytest_plugin.py`. Register via entry point `pytest11 = {"vectorlens" = "vectorlens.pytest_plugin"}`. Fixture auto-installs interceptors, runs test, collects groundedness.

**Ownership**: `vectorlens/pytest_plugin.py` (new), `tests/test_pytest_plugin.py`.

#### 2. GitHub Actions Regression Check
**What**: `.github/workflows/vectorlens.yml` template that runs RAG test suite, compares groundedness to baseline in `vectorlens.baseline.json`, comments on PRs if score drops >5%.

**Why it matters**: Prevents silent prompt regressions from reaching production. Developer sees "Groundedness regression: -7% (was 85%, now 78%)" right on the PR.

**How**: CLI commands: `vectorlens baseline` (saves current groundedness scores), `vectorlens check` (compares to baseline, exits non-zero if regression). GitHub Actions workflow calls these, posts comment on PR.

**Ownership**: `vectorlens/cli.py` (baseline/check commands), `.github/workflows/vectorlens.yml` (template in docs).

#### 3. SQLite Persistence
**What**: `vectorlens.serve(persist="./sessions.db")`. Sessions survive server restarts. `GET /api/sessions` returns historical + live.

**Why it matters**: Can revisit debugging sessions from last week. Historical trends visible.

**How**: Add optional `persist` kwarg to `serve()`. If provided, SessionBus saves sessions to SQLite on each update. On startup, load from DB. Dashboard shows "Cached" badge for historical sessions.

**Ownership**: `vectorlens/session_bus.py` (SQLite layer), `vectorlens/server/api.py` (add DB query endpoints).

#### 4. `vectorlens share` — Self-Contained Export
**What**: `vectorlens share <session-id>` generates a single HTML file with the full attribution view embedded (no server needed to view).

**Why it matters**: Easiest way to share debugging results with teammates. No "show me a screenshot". Email-able, Slack-uploadable, viewable offline.

**How**: New CLI command that serializes a session to JSON, bundles with React app (pre-built HTML), outputs single .html file. Can be opened in any browser.

**Ownership**: `vectorlens/cli.py` (share command), `vectorlens/server/export.py` (HTML generation), `dashboard/src/standalone.tsx` (React app for standalone mode).

#### 5. Token-Level Attribution (Research)
**What**: True per-subword attribution. Instead of "Chunk 3 contributed 65% to hallucination", we say "Chunk 3 word 'neural' contributed 12% to generated word 'quantum'".

**Why it matters**: Extreme precision. Especially valuable for understanding cross-attention failures.

**How**: For local models: extract attention weights from transformer. For API models: gradient approximation over prompt embeddings (expensive, opt-in). Requires research into efficient approaches.

**Ownership**: `vectorlens/attribution/token_level.py` (new), `vectorlens/detection/hallucination.py` (wiring).

---

### v1.0.0 — Production-Grade (Target: 6 months)

Features that make VectorLens worthy of being *the* RAG debugging standard.

#### 1. MCP Server Integration
**What**: `python -m vectorlens.mcp_server`. Claude Code, Cursor, VS Code extensions can query VectorLens: `@vectorlens What hallucinated in session abc123?`

**Why it matters**: Debugging seamlessly integrated into your IDE. Ask questions about your debugging sessions without leaving the editor.

**How**: Implement [Model Context Protocol](https://modelcontextprotocol.io/) server exposing VectorLens state as tools. Tools include: `query_session()`, `get_groundedness()`, `list_hallucinations()`.

**Ownership**: `vectorlens/mcp_server.py` (new), tests in `tests/test_mcp.py`.

#### 2. Multi-Project Dashboard
**What**: Track multiple RAG codebases in one VectorLens instance. Tabs per project. Groundedness trends over time.

**Why it matters**: DevOps teams managing multiple RAG services can monitor all of them from one dashboard.

**How**: Add `project_name` field to sessions. Dashboard allows filtering/switching between projects. `GET /api/projects` returns project list with aggregated metrics.

**Ownership**: `vectorlens/types.py` (add project_name field), `vectorlens/server/api.py` (project endpoints), `dashboard/src/components/ProjectSelector.tsx`.

#### 3. Plugin System
**What**: `[project.entry-points."vectorlens.interceptors"]` allows third-party interceptors. Community contributes support for Cohere, Together AI, Bedrock, custom vector DBs.

**Why it matters**: VectorLens doesn't need to add every provider. Community does.

**How**: `vectorlens/plugins.py` loads interceptors from entry points dynamically. `install_all()` includes both built-in and third-party interceptors.

**Ownership**: `vectorlens/plugins.py` (entry point loader), documentation in CONTRIBUTING.md.

#### 4. Embedding Model Comparison Mode
**What**: Compare attribution quality across embedding model choices. "Does text-embedding-3-large give better hallucination detection than text-embedding-3-small?"

**Why it matters**: Developers can benchmark embedding models on their own hallucinations—the real signal, not synthetic benchmarks.

**How**: Allow swapping detection model via `HallucinationDetector(model_name="...")`. Compare two runs side-by-side on the dashboard.

**Ownership**: `vectorlens/detection/hallucination.py` (model loading), `dashboard/src/components/ModelComparison.tsx`.

#### 5. Team Mode (Optional)
**What**: Optional auth (API key) for shared dashboards. Multiple developers, one VectorLens instance. Conversations shared automatically.

**Why it matters**: Teams can share a single debug instance for collaborative debugging.

**How**: Add optional password/API key auth. Sessions get `owner_id` field. Dashboard shows sessions only from your team.

**Ownership**: `vectorlens/server/auth.py` (new), `vectorlens/types.py` (add owner_id), `dashboard/src/components/Auth.tsx`.

---

## What We Will NOT Build

Be explicit and firm:

### No Cloud Version — Ever
Local-first is non-negotiable. The entire value proposition is "your data stays on your machine". A cloud version defeats that. If users want shared dashboards, they run VectorLens on a shared dev machine or expose localhost over SSH tunneling.

### No LLM Evaluation / Red-Teaming
That's what promptfoo, Agentive, LM-Harness do. We are not an evaluation framework. Different problem.

### No Training Data Platform
The annotation export in v0.2 is a feature (turn debugging into labeled data), not a product. We don't build versioning, de-duplication, active learning loops, or data governance. Use Argilla or Label Studio for that.

### No LLMOps Dashboard
We are not LangSmith. We are not Weights & Biases. We are not MLflow. We don't build:
- Prompt versioning / management
- Model performance tracking
- Cost tracking
- A/B testing frameworks
- Model routing
- Fine-tuning pipelines
- Production monitoring

All of those are different products. They deserve their own focus. We do one thing: hallucination debugging.

### No Paid Tier
MIT license means MIT license. No "open core" bait-and-switch where the good features are closed-source. Everything is MIT, forever.

---

## How to Contribute

### Best Contributions

1. **New interceptor** — Add support for Together AI, Cohere, AWS Bedrock, Mistral Web. Template: copy `openai_patch.py`, change ~30 lines. Takes an afternoon. High impact.

2. **Vector DB interceptor** — MongoDB Vector Search, Elasticsearch, Milvus, QdrantDB. Same template, different API.

3. **Dashboard improvements** — Streaming UI, A/B comparison view, annotation panel, model comparison view. React + TypeScript. Makes debugging faster.

4. **Documentation** — Tutorial notebooks, real-world examples (e.g., "debugging a multi-turn agent"), integration guides for specific frameworks.

5. **Bug reports** — Run VectorLens on your RAG project. If it breaks, file an issue with a minimal repro. This is how we find edge cases.

### What We Don't Want

- New features that require cloud accounts or data leaving the machine
- Increasing startup time or memory footprint (our target: <500MB total with model loaded, <100ms overhead per LLM call)
- Breaking the "zero instrumentation" contract (any feature requiring code changes to user RAG pipeline)
- Evaluation or red-teaming features (out of scope)

---

## How We Compare

| Tool | Cloud Required | Instrumentation | Hallucination Attribution | Streaming | Local-First |
|---|---|---|---|---|---|
| **VectorLens** | No | No | ✅ per-chunk | ✅ | ✅ |
| LangSmith | Optional (Docker) | Yes | Partial (traces only) | ✅ | Hybrid |
| Arize Phoenix | Optional | Yes (OTEL) | No | ✅ | Hybrid |
| RAGxplorer | No | Yes | No (viz only) | ❌ | ✅ |
| Langfuse | Optional | Yes | No | ✅ | Hybrid |

Our moat is crystal clear: **local-first + zero-instrumentation + causal attribution**. Everything else has at least one of these gaps. We own this space.

---

## Why VectorLens Matters

Most RAG debugging tools try to be *observability platforms*. They want to be your entire MLOps stack. They want cloud accounts. They want integrations everywhere.

VectorLens is different. We're a *debugging tool*. Like print debugging, like pytest, like a profiler. You install it, you run it locally, you understand what happened, you fix it. Then you move on.

That focus is our strength. We make hallucination debugging as fast as print debugging. That's the north star.

By v1.0, if you're building RAG in Python and you haven't tried VectorLens, you're doing it harder than you need to.
