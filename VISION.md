# VectorLens: Vision & Future Roadmap

**VectorLens** started as a lightweight, zero-telemetry, deep-attribution debugger for Retrieval-Augmented Generation (RAG) pipelines. As AI infrastructure matures, the need for fully local, transparent, and scalable tracing tools that do not lock developers into expensive cloud-based analytics platforms has never been higher.

This document outlines the short-term stabilization goals and the long-term strategic vision for VectorLens as it evolves into a production-grade observability engine for multi-agent systems and advanced LLM architectures.

---

## 1. Core Philosophy: The Developer First

*   **100% Local-First:** No API keys required for tracing. No telemetry sent to external servers. The developer maintains complete data sovereignty over their RAG sessions, logs, and attribution metrics.
*   **Zero-Config Integration:** The onboarding experience should remain entirely unobtrusive. No decorators, no massive code refactors, and no custom wrappers. `import vectorlens; vectorlens.serve()` will remain the standard.
*   **Performance Without Compromise:** Tracing and attribution should never become the bottleneck. VectorLens targets near-zero latency overhead for the host application by offloading compute-intensive attribution (like SentenceTransformers and Perturbation) to background threads.

---

## 2. Short-Term Resolution Arc (v0.2.x -> v0.3.x)

The immediate focus is hardening the core engine, specifically addressing edge case memory spikes, framework interoperability, and stream limits found during security sweeps.

### Stability & Resilience
- **Context Manager Compatibility:** Immediate fixes to ensure `__enter__` and `__aenter__` proxying for libraries like `httpx` to prevent unexpected crashes in target applications during streaming.
- **Memory Footprint Bounds:** Implementing strict chunk limits/truncation limits for local embedding processes (SentenceTransformer) to prevent Out-Of-Memory (OOM) situations on giant, runaway hallucinated outputs.
- **Persistent Sessions (Opt-In SQLite):** Moving from a purely volatile RAM session store (capped at 50-200 sessions) to a durable local SQLite store. This prevents developer laptops from bloating over multi-day debugging sessions and allows historical comparisons.

### Deeper Attribution Mechanisms
- **Conditional Attribution:** Only run the heavy lifting (Attention Rollout / Perturbation) when basic semantic validation flags a potential hallucination, saving local compute time.
- **LIME-Style Perturbation as a Standard:** Exposing the new N+1 (Ridge Regression) perturbation APIs gracefully in the UI for users to "deep inspect" specific chunks.

---

## 3. Mid-Term Architectural Expansion (v0.5.x+)

As the local engine solidifies, VectorLens will expand beyond basic RAG to support complex, multi-hop agentic architectures.

### Framework-Agnostic Agent Tracing (AgentReplay Integration)
- **LangChain / LlamaIndex Native Support:** While monkey-patching `httpx` and `openai` catches network traffic, we need DAG-aware tracing for frameworks like LangChain. VectorLens will map the full execution graph (Thought -> Tool -> Observation).
- **Time-Travel Debugging:** Moving beyond single-shot attribution to an "AgentReplay"-style dashboard where developers can scrub backward through an agent's memory state to see exactly what prompted a failure 10 steps prior.

### Advanced Observability
- **Token-Level Granularity:** Transitioning from sentence-level cosine similarity to exact-token hallucination highlights using internal attention rollouts.
- **Cross-Session Prompt A/B Testing:** Providing a dedicated diff view where developers can compare Groundedness Scores, Cost, and Latency between two iterations of their prompts across an identical dataset.

---

## 4. Long-Term Vision: The Open Standard for AI Execution States

Ultimately, VectorLens aims to establish a universal, open standard format for serialized AI execution graphs.

If VectorLens can standardize how a single LLM request (its context window, retrieved chunk payload, tool payload, and probability matrices) is stored locally as an immutable "Node", then the ecosystem can build generic tooling around this standard. The vision is for VectorLens to be the engine that silently catches, stores, and organizes the chaotic inner consciousness of your AI systems, providing unparalleled clarity when things go wrong.
