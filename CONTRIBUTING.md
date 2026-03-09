# Contributing to VectorLens

Welcome! We're excited that you're interested in contributing. VectorLens is a community-driven project, and we value contributions of all kinds — code, documentation, bug reports, and feature ideas.

This guide will help you get started quickly.

---

## Quick Start for Contributors

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/vectorlens.git
cd vectorlens
```

### 2. Install in Development Mode

```bash
pip install -e ".[all]"
```

This installs VectorLens locally with all optional dependencies (OpenAI, Anthropic, ChromaDB, etc.) so you can test against all integrations.

### 3. Install Dashboard Dependencies

```bash
cd dashboard
npm install
```

### 4. Run Tests

```bash
# Fast unit tests (no ML models, ~5s)
pytest tests/ -m "not integration"

# Full integration tests with sentence-transformers (slow, ~30s)
pytest tests/ -m integration
```

All tests should pass before submitting a PR.

---

## Project Structure Overview

Understanding the codebase makes contributing easier:

```
vectorlens/
├── interceptors/          # Monkey-patches for LLM clients & vector DBs
│   ├── openai_patch.py   # OpenAI integration
│   ├── anthropic_patch.py
│   ├── gemini_patch.py
│   ├── chroma_patch.py   # ChromaDB integration
│   ├── pinecone_patch.py
│   ├── faiss_patch.py
│   ├── weaviate_patch.py
│   └── transformers_patch.py
│
├── detection/            # Hallucination detection
│   └── hallucination.py  # Sentence-transformers embedding + cosine similarity
│
├── attribution/          # Chunk attribution
│   └── perturbation.py   # Optional: drop chunks, measure divergence
│
├── session_bus.py        # Thread-safe in-process event bus
├── pipeline.py           # Auto-attribution pipeline (background thread)
├── types.py              # Shared dataclasses (events, sessions)
│
└── server/
    ├── app.py            # FastAPI app definition
    ├── api.py            # REST & WebSocket endpoints
    └── static/           # React dashboard (built from dashboard/dist/)

dashboard/                # React + TypeScript frontend
├── src/components/       # UI components
├── src/hooks/           # Custom React hooks
├── src/App.tsx
├── package.json
└── tsconfig.json

tests/                    # Unit & integration tests
├── conftest.py         # Test fixtures & mocks
├── test_detection.py
├── test_attribution.py
└── test_interceptors/
```

**Key concept**: Everything flows through the **SessionBus** event stream. Interceptors publish events → Pipeline subscribes → Server API streams to Dashboard via WebSocket.

---

## How to Add a New LLM Interceptor

Let's say you want to add support for a new LLM provider (e.g., Together AI).

### Step 1: Create the Patch File

Create `vectorlens/interceptors/together_patch.py`:

```python
"""Together AI LLM interceptor."""
from __future__ import annotations

import functools
import time
from typing import Any

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import LLMRequestEvent, LLMResponseEvent


class TogetherInterceptor(BaseInterceptor):
    """Intercepts Together AI API calls."""

    def __init__(self):
        self._installed = False
        self._original_create = None

    def install(self):
        """Patch the Together client."""
        if self._installed:
            return

        try:
            import together
        except ImportError:
            return  # Library not installed, skip silently

        # Patch the client method
        self._original_create = together.client.Chat.create
        together.client.Chat.create = self._wrap_create(self._original_create)
        self._installed = True

    def uninstall(self):
        """Restore the original client method."""
        if not self._installed:
            return

        try:
            import together

            if self._original_create:
                together.client.Chat.create = self._original_create
        except ImportError:
            pass

        self._installed = False

    def is_installed(self) -> bool:
        """Check if interceptor is active."""
        return self._installed

    def _wrap_create(self, original):
        """Wrap the create method to intercept calls."""

        @functools.wraps(original)
        def wrapper(self_, **kwargs):
            # Extract LLM parameters
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])

            # Record request
            request_event = LLMRequestEvent(
                provider="together",
                model=model,
                messages=messages,
            )
            bus.record_llm_request(request_event)

            # Call original method
            start = time.time()
            response = original(self_, **kwargs)
            latency_ms = (time.time() - start) * 1000

            # Extract response details
            output_text = response.choices[0].message.content
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

            # Record response
            response_event = LLMResponseEvent(
                request_id=request_event.id,
                output_text=output_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            bus.record_llm_response(response_event)

            return response

        return wrapper
```

### Step 2: Register in `__init__.py`

Edit `vectorlens/interceptors/__init__.py` and add:

```python
from vectorlens.interceptors.together_patch import TogetherInterceptor

_INTERCEPTORS["together"] = TogetherInterceptor()
```

### Step 3: Add Tests

Create `tests/test_interceptors/test_together_patch.py`:

```python
"""Tests for Together AI interceptor."""
import pytest
from unittest.mock import MagicMock, patch

from vectorlens.interceptors import install_all, uninstall_all


def test_together_install():
    """Test Together interceptor installation."""
    with patch("vectorlens.interceptors.together_patch.together"):
        install_all()
        # Verify the patch is applied
        uninstall_all()


def test_together_not_installed_gracefully():
    """Test that missing Together library doesn't break."""
    with patch.dict("sys.modules", {"together": None}):
        install_all()  # Should not raise
        uninstall_all()
```

### Step 4: Update pyproject.toml (Optional)

If your provider requires external dependencies, add an optional dependency group:

```toml
[project.optional-dependencies]
together = ["together-ai>=1.0.0"]
```

---

## How to Add a New Vector DB Interceptor

Same pattern as LLM interceptors. Example: adding support for Elasticsearch.

Create `vectorlens/interceptors/elasticsearch_patch.py`:

```python
"""Elasticsearch vector DB interceptor."""
from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import VectorQueryEvent, RetrievedChunk


class ElasticsearchInterceptor(BaseInterceptor):
    def __init__(self):
        self._installed = False
        self._original_search = None

    def install(self):
        try:
            from elasticsearch import Elasticsearch
        except ImportError:
            return

        self._original_search = Elasticsearch.search
        Elasticsearch.search = self._wrap_search(self._original_search)
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        try:
            from elasticsearch import Elasticsearch

            if self._original_search:
                Elasticsearch.search = self._original_search
        except ImportError:
            pass
        self._installed = False

    def is_installed(self) -> bool:
        return self._installed

    def _wrap_search(self, original):
        def wrapper(self_, *args, **kwargs):
            # Call original
            response = original(self_, *args, **kwargs)

            # Extract query text (usually in body parameter)
            query_text = kwargs.get("query", {}).get("match", {}).get("content", "")

            # Extract results
            results = []
            for hit in response["hits"]["hits"]:
                results.append(
                    RetrievedChunk(
                        chunk_id=hit["_id"],
                        text=hit["_source"]["content"],
                        score=hit["_score"],
                        metadata=hit["_source"].get("metadata", {}),
                    )
                )

            # Record event
            event = VectorQueryEvent(
                db_type="elasticsearch",
                collection=kwargs.get("index", "default"),
                query_text=query_text,
                results=results,
            )
            bus.record_vector_query(event)

            return response

        return wrapper
```

Then register in `interceptors/__init__.py`:

```python
from vectorlens.interceptors.elasticsearch_patch import ElasticsearchInterceptor

_INTERCEPTORS["elasticsearch"] = ElasticsearchInterceptor()
```

---

## Running Tests

### Unit Tests (Fast, Recommended for Development)

```bash
pytest tests/ -m "not integration" -v
```

- Mocks `sentence-transformers` model
- No ML model downloads
- Fast feedback loop (~5s)
- Great for CI/CD

### Integration Tests (Full, Before Release)

```bash
pytest tests/ -m integration -v
```

- Loads real `sentence-transformers` model (~100MB auto-download)
- Tests actual hallucination detection
- Slower (~30s)
- Run before releasing to main

### Specific Test Modules

```bash
pytest tests/test_detection.py -v
pytest tests/test_attribution.py -v
pytest tests/test_interceptors/ -v
```

### With Coverage Report

```bash
pytest tests/ --cov=vectorlens --cov-report=html
open htmlcov/index.html
```

---

## Dashboard Development

### Building the Dashboard

```bash
cd dashboard
npm run build
```

This generates optimized production files in `dashboard/dist/`, which the server bundles into the wheel.

### Development Mode

```bash
cd dashboard
npm run dev
```

- Starts Vite dev server on `http://localhost:5173`
- Hot module reloading (instant feedback on changes)
- Proxies API requests to backend on port 7756
- Great for iterating on UI components

### Dashboard Stack

- **Framework**: React 18 + TypeScript
- **Styling**: Tailwind CSS
- **Build**: Vite
- **State**: React Context + hooks (no Redux)

---

## Code Style & Type Hints

### Formatting

We use **black** + **isort** for consistent formatting.

```bash
# Auto-format all Python files
black vectorlens tests
isort vectorlens tests
```

Most pre-commit hooks will run these automatically.

### Type Hints

**Required for all new functions and classes:**

```python
from typing import Optional

def detect_hallucinations(
    output_text: str,
    chunks: list[RetrievedChunk],
) -> list[OutputToken]:
    """Detect hallucinated sentences using semantic similarity."""
    pass
```

Avoid `Any` except where truly unavoidable. If you need it, add a comment explaining why:

```python
def flexible_call(client: Any) -> Any:  # Accept any LLM client type
    pass
```

---

## Pull Request Checklist

Before submitting a PR:

- [ ] **Tests pass**: Run `pytest tests/ -m "not integration"` locally
- [ ] **Type hints**: All new functions have type hints
- [ ] **Code style**: Run `black` and `isort`
- [ ] **New interceptor?** Registered in `interceptors/__init__.py`
- [ ] **Dashboard changes?** Run `npm run build` and commit the updated dist folder
- [ ] **CHANGELOG updated**: Added entry to `CHANGELOG.md` under `[Unreleased]`
- [ ] **No new required dependencies**: Optional deps are fine (add to `pyproject.toml`)
- [ ] **Commit message**: Follows [Conventional Commits](https://www.conventionalcommits.org/)

Example commit messages:

```
feat: add Together AI LLM interceptor
fix: handle empty chunk list in detection
refactor: simplify sentence splitting logic
docs: add pgvector integration example
test: add coverage for Weaviate interceptor
```

---

## Marking Issues as "Good First Issues"

Help onboard new contributors by labeling beginner-friendly tasks:

**What makes a good first issue:**

- Scoped to a single module (not cross-cutting)
- Doesn't require deep familiarity with ML models
- Has clear acceptance criteria
- Can be completed in 2-4 hours
- Includes pointers to relevant code

**Examples:**

- "Add tests for X interceptor"
- "Improve error message when Y fails"
- "Add type hints to legacy function Z"
- "Write docstring for module A"

---

## Questions & Discussions

Have questions? Don't open an issue — use **GitHub Discussions** instead!

- **Bug reports** → [Issues](https://github.com/YOUR_REPO/vectorlens/issues)
- **Feature ideas** → [Discussions](https://github.com/YOUR_REPO/vectorlens/discussions)
- **How-to questions** → [Discussions](https://github.com/YOUR_REPO/vectorlens/discussions)

This keeps Issues focused on actionable work.

---

## Getting Help

### Debugging Tips

1. **Check existing issues**: Your question might already be answered
2. **Read CLAUDE.md**: Architecture notes and gotchas
3. **Look at tests**: Tests show how to use the API
4. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

### Common Tasks

- **Adding a new LLM provider**: See "How to Add a New LLM Interceptor" above
- **Changing detection algorithm**: Edit `vectorlens/detection/hallucination.py`
- **Understanding the event flow**: Trace through `session_bus.py` → `pipeline.py` → `server/api.py`
- **Customizing the dashboard**: Edit files in `dashboard/src/`

---

## Recognition

Contributors are recognized in:

- **CHANGELOG.md** (for releases)
- **GitHub Contributors** page
- Project README (for major contributions)

---

## Code of Conduct

Please read our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). We're committed to providing a welcoming, harassment-free environment.

---

## Thank You!

Your contributions make VectorLens better for everyone. We're grateful for your time and effort. If you have questions or get stuck, don't hesitate to ask — we're here to help!

Happy coding!
