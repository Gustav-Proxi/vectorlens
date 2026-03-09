# VectorLens Roadmap

## Vision

VectorLens is the local-first, zero-config observability layer for RAG pipelines. The goal is to make hallucination debugging as fast as print debugging — no cloud account, no SDK rewrites, just `import vectorlens; vectorlens.serve()`. By v1.0, VectorLens should be the default RAG debugger the way pytest is the default Python test runner — everyone installs it without thinking about it.

## Design Principles

- **Local-first, forever**: All data stays on the developer's machine. No cloud, no external APIs, no data leaving the process.
- **Zero config, zero friction**: Three lines of code. Works with existing RAG pipelines unchanged. No wrapper functions, no prompt rewrites.
- **Transparent to the user's code**: Monkey-patching interception at the transport layer. If you refactor your RAG code, VectorLens adapts automatically.
- **Performance is non-negotiable**: Background attribution pipeline never blocks. Users see results in milliseconds, not seconds.
- **One tool, one job**: Hallucination debugging and attribution. We don't build evaluation, red-teaming, fine-tuning, or cloud dashboards. Stay focused.

## v0.2.0 — The Developer Loop (target: 4-6 weeks)

Make the day-to-day debugging loop tight. The MVP (v0.1) detects hallucinations; v0.2 makes it *fast* to iterate and *obvious* which changes matter.

- **Real-time streaming dashboard** (issue #13) — As tokens stream in, the output panel updates live via WebSocket. Partial `LLMResponseEvent` pushes show "Analyzing..." state, replaced by final attribution when complete. Developers see results as they type queries.

- **`vectorlens.assert_grounded()` inline assertion** (issue #14) — `from vectorlens.testing import assert_grounded; assert_grounded(response, chunks, min_score=0.8)`. Works in pytest, Jupyter notebooks, standalone scripts. Fail fast on low-quality RAG.

- **Prompt A/B comparison mode** (issue #15) — Select two sessions in sidebar → side-by-side groundedness comparison showing `+12% groundedness`, `-2 hallucinated sentences`. Perfect for iterating on prompt wording and understanding what sticks.

- **Human annotation layer** (issue #16) — Right-click a hallucinated span → mark as "actually correct" / "confirmed hallucination" / "unsure". Annotations persist in localStorage. Export as labeled dataset for fine-tuning. Developers crowdsource ground truth.

## v0.3.0 — Team & CI (target: 8-12 weeks)

Scale from solo debugging to team collaboration and automated testing. Catch quality regressions before they ship.

- **pytest plugin with auto-assertions** (issue #17) — `pip install vectorlens[pytest]` adds a `vectorlens_session` fixture. `@pytest.mark.rag` decorator auto-asserts `session.overall_groundedness >= threshold`. Entry point: `pytest11 = {"vectorlens" = "vectorlens.pytest_plugin"}`. CI/CD now tracks hallucination metrics alongside unit tests.

- **GitHub Actions groundedness regression check** (issue #18) — `.github/workflows/vectorlens.yml` template that runs RAG test suite, compares groundedness to baseline in `vectorlens.baseline.json`, comments on PRs if score drops >5%. "Groundedness regression: -7% (was 85%, now 78%)" right on the PR.

- **SQLite session persistence** (issue #19) — `vectorlens.serve(persist="./sessions.db")`. Sessions survive server restarts. `GET /api/sessions` returns historical + live. Dashboard shows "Cached" badge on historical sessions. Developers can review old debugging sessions weeks later.

- **`vectorlens share` self-contained export** (issue #20) — `vectorlens share <session-id>` generates a single HTML file with the full attribution view embedded (no server needed to view). Share as email attachment, Slack upload, or GitHub issue. Async collaboration without screenshots.

## v1.0.0 — Production-Grade (target: 6 months)

Make it worthy of being the default RAG debugging tool. The features that matter for enterprise deployments and complex RAG systems.

- **MCP server integration** — Claude Code, Cursor, and VS Code extensions can query VectorLens mid-session: `@vectorlens What hallucinated in session abc123?` Debugging seamlessly integrated into the IDE.

- **Multi-project dashboard** — Track multiple RAG codebases in one VectorLens instance. Switch between projects, compare attribution across different services.

- **Team authentication (optional)** — Password protection for shared dashboards. Only relevant when multiple team members access the same VectorLens instance.

- **Attention-based attribution for local models** — Token-level attribution for HuggingFace models without perturbation cost. Extract attention weights from transformer layers, show exact word-level hallucinations.

- **Embedding model comparison** — Benchmark different embedding models (all-MiniLM, bge-m3, E5, etc.) on the same hallucinations. "This hallucination is detected by E5 but not MiniLM — choose model accordingly."

- **Streaming response support (full)** — Correctly attribute streamed token sequences. Dashboard updates as chunks arrive from LLM.

- **Plugin system** — Custom attribution algorithms via Python entry points. Users can plug in their own hallucination detectors, perturbation strategies, or scoring methods.

- **Grafana export** — Sessions as time-series metrics (groundedness%, hallucination_count) pushed to Prometheus or InfluxDB for team dashboards.

## What We Will NOT Build

- **Cloud-hosted version**: Stay local-first forever. If users want shared dashboards, they run VectorLens on a shared dev machine or expose localhost over SSH tunneling. No SaaS, no subscription, no data exfiltration.

- **LLM evaluation & red-teaming**: That's what promptfoo and other tools do. VectorLens does attribution, not evaluation. Orthogonal problems.

- **Full LLMOps platform**: We don't build prompt management, versioning, A/B testing frameworks, cost tracking, model routing, or fine-tuning pipelines. That's too much scope. We do one thing well: *Why did your RAG hallucinate?*

- **Training data curation & labeling**: Different workflow (annotation loops, dataset versioning). Use Argilla, Label Studio, or Cord for that.
