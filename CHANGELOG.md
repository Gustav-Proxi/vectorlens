# Changelog

All notable changes to VectorLens are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
