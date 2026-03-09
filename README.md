# VectorLens

> **See *why* your RAG hallucinates** — token-level attribution in 30 seconds, zero config, no cloud.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

When your RAG pipeline hallucinates, you're left guessing. Which chunk caused it? Did the retriever fail? Did the LLM ignore the context? VectorLens answers these questions in real-time with a local dashboard — no signup, no data leaving your machine.

<!-- Dashboard screenshot: dark theme showing LLM output with hallucinated sentences highlighted in red,
     below: retrieved chunks panel with attribution scores, groundedness meter at top (72% green). -->

## The Problem

**RAG debugging is painful.** When an LLM hallucinates, you have:
- Logs showing *what* it said — but not *why*
- Retrieved chunks — but no way to know which caused the hallucination
- Existing tools (LangSmith, Arize Phoenix) require cloud accounts, vendor lock-in, and dashboard learning curves

**The gap**: No local tool that connects the dots between retrieved context and LLM output with zero instrumentation.

## The Solution

**Three lines of code. Local dashboard. Instant answers.**

```python
import vectorlens
vectorlens.serve()  # http://127.0.0.1:7756

# Your RAG code — completely unchanged
results = collection.query(query_texts=["Q"], n_results=5)
response = client.chat.completions.create(model="gpt-4o", messages=[...])
# Dashboard now shows which chunks caused each hallucination
```

**What you get**:
- Hallucinated sentences highlighted in red
- Which retrieved chunks caused each hallucination
- Attribution scores (0–100%) per chunk
- Overall groundedness percentage
- Real-time updates via WebSocket
- All running locally — zero external dependencies

---

## How It Works

```
Your RAG code (unchanged)
    │
    ▼
[VectorLens Interceptors]─────────────┐
 OpenAI │ Anthropic │ Gemini │       │
 ChromaDB │ Pinecone │ FAISS │       │
 pgvector │ LangChain │ ...         │
    │                              │
    ▼                              │
[Session Bus]                       │
 In-process event stream            │
 (no external services)             │
    │                              │
    ▼                              │
[Attribution Pipeline]              │
 1. Detect hallucinated sentences  │
 2. Score chunk contributions      │
 3. Compute overall groundedness   │
    │                              │
    ▼                              │
[FastAPI Server] ◄─────────────────┘
    │
    ▼
[React Dashboard]
 http://127.0.0.1:7756
```

**The flow**:

1. **Interception** (transparent): Patches LLM client methods and vector DB calls. Zero code changes required.

2. **Event Bus** (in-process): Every query and response flows through a thread-safe event bus. No network overhead, no external services.

3. **Attribution Pipeline** (background): Runs hallucination detection and chunk scoring in parallel. Never blocks your main RAG code.

4. **Dashboard** (local): React UI shows results in real-time via WebSocket.

**Key Details**:
- Uses `sentence-transformers/all-MiniLM-L6-v2` (22MB, CPU-only, no API calls)
- Cosine similarity threshold 0.4 to detect hallucinations (conservative)
- Attribution scores computed from top-3 chunks per sentence
- Conditional attribution: skip deep analysis if response is fully grounded (~50ms vs ~500ms savings)

---

## Installation

### Base install

```bash
pip install vectorlens
```

### With your LLM provider (choose one or more)

```bash
pip install "vectorlens[openai]"        # OpenAI (GPT-4o, etc.)
pip install "vectorlens[anthropic]"     # Anthropic (Claude)
pip install "vectorlens[gemini]"        # Google Gemini
pip install "vectorlens[mistral]"       # Mistral
```

### With your vector database

```bash
pip install "vectorlens[chromadb]"      # ChromaDB
pip install "vectorlens[pinecone]"      # Pinecone
pip install "vectorlens[faiss]"         # FAISS
pip install "vectorlens[weaviate]"      # Weaviate
```

### Everything at once

```bash
pip install "vectorlens[all]"
```

**Requirements**: Python 3.11+. Minimal dependencies: FastAPI, uvicorn, sentence-transformers, httpx.

---

## Quickstart

### Example 1 — OpenAI + ChromaDB (Most Common)

```python
import vectorlens

# Start the dashboard
vectorlens.serve()  # http://127.0.0.1:7756

# Your existing RAG code — unchanged
import chromadb
import openai

collection = chromadb.Client().get_or_create_collection("docs")
client = openai.OpenAI()

# Load documents
collection.add(
    ids=["id1", "id2", "id3"],
    documents=[
        "Attention is a mechanism allowing models to focus on relevant input parts.",
        "The transformer uses self-attention to compute token relationships.",
        "BERT is a large language model trained on masked language modeling."
    ]
)

# Query
results = collection.query(
    query_texts=["How does attention work?"],
    n_results=3
)

# Build context
context = "\n".join(results["documents"][0])

# LLM call
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: How does attention work?"
        }
    ]
)

print(response.choices[0].message.content)
# Dashboard shows:
# - Retrieved chunks and similarity scores
# - Which sentences are hallucinated
# - Which chunks explain which sentences
# - Overall groundedness %
```

Visit http://127.0.0.1:7756 in your browser while the script runs.

### Example 2 — Anthropic + Streaming

```python
import vectorlens
vectorlens.serve()

import anthropic

client = anthropic.Anthropic()

# Streaming works automatically
with client.messages.stream(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Answer this question based on the context..."
        }
    ]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

# Full response reconstructed and attributed after stream ends
```

### Example 3 — LangChain RetrievalQA

```python
import vectorlens
vectorlens.serve()

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# LangChain chain — VectorLens intercepts BaseChatModel + BaseRetriever
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=Chroma(...).as_retriever(),
    return_source_documents=True
)

result = qa.invoke("What is self-attention?")
# Both retriever results and LLM call captured, linked, attributed
```

### Example 4 — Custom Vector DB (Manual API)

For unsupported vector databases, manually record events:

```python
import vectorlens
from vectorlens.session_bus import bus
from vectorlens.types import VectorQueryEvent, RetrievedChunk

vectorlens.serve()

# Your custom vector DB search (pgvector, Elasticsearch, Milvus, etc.)
async def my_pgvector_search(query: str) -> list[dict]:
    results = await pgvector_client.search(query, limit=5)

    # Manually record vector query
    event = VectorQueryEvent(
        db_type="pgvector",
        collection="documents",
        query_text=query,
        results=[
            RetrievedChunk(
                chunk_id=str(r["id"]),
                text=r["text"],
                score=r["similarity_score"],
                metadata={"source": r.get("source")}
            )
            for r in results
        ]
    )
    bus.record_vector_query(event)
    return results

# Use in your RAG pipeline
result = my_pgvector_search("What is quantum computing?")
# Dashboard shows attribution to pgvector results
```

---

## Supported Integrations

| Provider | Type | What's Captured | Status |
|---|---|---|---|
| **OpenAI** | LLM | model, messages, tokens, cost, latency | ✓ Native |
| **Anthropic** | LLM | same + streaming | ✓ Native |
| **Google Gemini** | LLM | same + SDK v1 & v2 | ✓ Native |
| **Mistral** | LLM | same | ✓ Native |
| **HuggingFace** | LLM | model, tokens, latency + attention weights | ✓ Native |
| **LangChain** | Framework | LLM calls, retrievers, chain steps | ✓ Native |
| **ChromaDB** | Vector DB | similarity scores, metadata | ✓ Native |
| **Pinecone** | Vector DB | same + namespace | ✓ Native |
| **FAISS** | Vector DB | L2/dot-product distances | ✓ Native |
| **Weaviate** | Vector DB | similarity scores, metadata | ✓ Native |
| **pgvector** | Vector DB | SQL similarity operators (`<=>`, `<->`, `<#>`) | ✓ Native |
| **Custom DB** | Any | via manual event API | ✓ See Example 4 |

**Cost Calculation** (as of March 2026):
- GPT-4o: $5/1M input tokens, $15/1M output tokens
- Claude 3.5 Sonnet: $3/1M input, $15/1M output
- Gemini 2.0 Flash: $0.075/1M input, $0.30/1M output

---

## Dashboard Tour

### Session Sidebar (Left)
- **Live Session** (green dot): Current active session with real-time event updates
- **Session History** (📋): Previously recorded sessions, cached in localStorage, survive server restarts
- Click any session to view its attribution results

### Groundedness Meter (Top)
- **0–100%** scale: Percentage of output grounded in retrieved chunks
- **Color coding**: Red (0–30%) = hallucinated, Yellow (30–70%) = mixed, Green (70–100%) = grounded

### LLM Output Panel (Center)
- Full response text with sentence-level highlighting
  - **Red background**: Hallucinated (cosine similarity < 0.4 to all chunks)
  - **Normal text**: Grounded in retrieved chunks
- **Hover any sentence**: See contributing chunks with attribution percentages
- Metadata: token counts, latency (ms), estimated cost (USD)

### Retrieved Chunks Panel (Right)
- Sorted by attribution score (highest to lowest)
- Each chunk shows:
  - **Similarity**: Original vector DB score
  - **Attribution %**: How much this chunk influenced the output
  - **Metadata**: Custom fields (source, page number, etc.)
  - **⚠️ Badge**: Contributed to hallucination
- Click to highlight related sentences in output

---

## Streaming Support

VectorLens captures streaming responses (`stream=True`). SSE chunks are intercepted at the httpx transport layer, and the full text is reconstructed after the stream completes.

**Works with**:
- OpenAI (streaming)
- Anthropic (streaming)
- Google Gemini (streaming)
- Mistral (streaming)

**Note**: Streaming responses estimate completion tokens from word count (exact counts unavailable in SSE chunks).

---

## Multi-turn Agents & Conversation DAGs

VectorLens tracks multi-turn conversation DAGs via `parent_request_id` linking. For multi-turn agents:

```python
import vectorlens

conversation_id = vectorlens.bus.start_conversation()
# All sessions in this conversation share conversation_id
# LLM calls are linked via parent_request_id → visible as connected tree
```

Example: LangChain agent automatically linked via parent_request_id:

```python
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

agent = create_openai_functions_agent(llm=ChatOpenAI(), ...)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({"input": "What is the capital of France?"})
# Dashboard shows: user query → agent call → tool call → agent response
```

---

## API Reference

### `vectorlens.serve()`

Start the dashboard server and install interceptors.

```python
url = vectorlens.serve(
    host: str = "127.0.0.1",      # Bind address
    port: int = 7756,             # Dashboard port
    open_browser: bool = True,    # Auto-open browser
    auto_intercept: bool = True   # Auto-install interceptors
) -> str
```

**Returns**: Dashboard URL string (e.g., `"http://127.0.0.1:7756"`)

**Behavior**:
- Starts FastAPI server on separate daemon thread (non-blocking)
- Returns immediately
- Auto-installs interceptors for installed libraries
- Opens browser if `open_browser=True`
- Logs warning if already running

**Example**:
```python
import vectorlens
url = vectorlens.serve(port=7756, open_browser=True)
print(f"Dashboard: {url}")
```

### `vectorlens.new_session()`

Start a fresh tracing session. Useful for isolating test runs.

```python
session_id = vectorlens.new_session() -> str
```

**Returns**: New session ID (UUID string)

**Behavior**:
- Creates blank Session object
- Sets it as active session (all new events go here)
- Previous session remains accessible in history

**Example**:
```python
# Test 1
sess1 = vectorlens.new_session()
# Run first test...

# Test 2
sess2 = vectorlens.new_session()
# Run second test...

# Both visible in dashboard history
```

### `vectorlens.stop()`

Shutdown the dashboard and uninstall all interceptors.

```python
vectorlens.stop() -> None
```

**Behavior**:
- Stops FastAPI server
- Uninstalls all monkey-patches
- Restores original client methods

**Example**:
```python
try:
    vectorlens.serve()
    # ... run RAG code ...
finally:
    vectorlens.stop()  # Clean shutdown
```

### `vectorlens.get_session_url()`

Get the dashboard URL for a specific session.

```python
url = vectorlens.get_session_url(
    session_id: str | None = None
) -> str
```

**Returns**: Dashboard URL for the session

**Example**:
```python
sess_id = vectorlens.new_session()
url = vectorlens.get_session_url(sess_id)
print(f"View session: {url}")
```

---

## How Attribution Works

### Hallucination Detection (Sentence-Level)

VectorLens detects hallucinations by comparing semantic similarity:

**Algorithm**:
1. Split LLM output into sentences
2. Embed each sentence using `all-MiniLM-L6-v2` (384-dimensional, CPU-only)
3. Embed each retrieved chunk with the same model
4. Compute cosine similarity between sentence and all chunks
5. **Detect as hallucinated** if `max(similarities) < 0.4`
6. For grounded sentences, record top-3 most similar chunks as attribution weights

**Threshold Justification**: 0.4 cosine similarity is conservative:
- Avoids false positives (marking correct sentences as hallucinated)
- Works well empirically with sentence-transformers' semantic space
- Tuned on common RAG failure cases

**Conditional Attribution**: Deep attribution (perturbation or attention rollout) only runs when hallucinations detected. Grounded responses skip expensive analysis (~50ms vs ~500ms savings).

**Attribution Methods**:
- **LIME perturbation** (API models): K=7 random binary masks over chunks. Ridge regression fits mask vectors to output similarity scores. Cost: exactly 7 LLM calls regardless of chunk count.
- **Attention rollout** (local HuggingFace models): Extracts token-level attention weights from model internals. Zero extra LLM calls.

**Example**:
```
Output: "Transformers use self-attention to compute token relationships."
Chunk 1: "The transformer architecture uses self-attention to compute relationships..."
         Similarity: 0.85 ✓ GROUNDED

Chunk 2: "BERT was trained on Wikipedia data."
         Similarity: 0.15 ✗ NOT RELEVANT

Max similarity: 0.85 > 0.4 → Sentence is GROUNDED
```

**Limitations**:
- Sentence-level, not token-level (token-level coming in v0.2)
- Hallucinated phrase within a grounded sentence may not be flagged
- May miss subtle semantic errors (swapped facts)

---

## Architecture

```
vectorlens/
├── __init__.py              Public API: serve(), stop(), new_session()
├── types.py                 Shared dataclasses (Session, LLMRequestEvent, etc.)
├── session_bus.py           In-process event bus with ContextVar isolation
├── pipeline.py              Auto-attribution with bounded ThreadPoolExecutor
│
├── interceptors/            Patches for every major LLM SDK + vector DB
│   ├── httpx_transport.py   Transport layer (OpenAI, Anthropic, Gemini, Mistral)
│   ├── openai_patch.py      OpenAI SDK (fallback)
│   ├── anthropic_patch.py   Anthropic SDK (fallback)
│   ├── gemini_patch.py      Google Gemini (fallback)
│   ├── langchain_patch.py   LangChain framework
│   ├── chroma_patch.py      ChromaDB
│   ├── pinecone_patch.py    Pinecone
│   ├── faiss_patch.py       FAISS
│   ├── weaviate_patch.py    Weaviate
│   ├── pgvector_patch.py    SQLAlchemy pgvector
│   └── transformers_patch.py HuggingFace pipelines
│
├── detection/               Hallucination detection
│   └── hallucination.py     Sentence-transformers embedding + similarity
│
├── attribution/             Chunk attribution methods
│   ├── perturbation.py      LIME-style bounded perturbation
│   └── attention.py         Attention rollout for local models
│
└── server/                  FastAPI + React dashboard
    ├── app.py               FastAPI app, ASGI middleware, WebSocket
    ├── api.py               REST endpoints
    └── static/              React dashboard (built dist files)

dashboard/                   React + TypeScript UI
├── src/
│   ├── components/          UI components
│   ├── hooks/               Custom hooks (useWebSocket, etc.)
│   └── App.tsx              Main app component
└── package.json
```

**Key Design Patterns**:

1. **Monkey-patching**: Directly patches client library methods. Zero wrapper overhead, zero code changes.
2. **Event bus**: SessionBus publishes events, pipeline subscribes. Loose coupling.
3. **Non-blocking attribution**: Background daemon thread. Main code never waits.
4. **Lazy-loaded models**: Sentence-transformers loads on first hallucination detection, not on import.
5. **Thread-safe sessions**: All state in SessionBus with locks. Safe for multithreaded usage.

---

## Running Tests

### Unit Tests (Fast, No ML Models)

```bash
python -m pytest tests/ -m "not integration"
```

- Mocks sentence-transformers
- Fast (< 5 seconds)
- Deterministic
- Good for CI/CD

### Integration Tests (Full, With ML Models)

```bash
python -m pytest tests/ -m integration
```

- Loads real sentence-transformers model
- Slow (30+ seconds)
- Tests actual hallucination detection
- Good for pre-release validation

### Specific Modules

```bash
python -m pytest tests/test_detection.py -v
python -m pytest tests/test_attribution.py -v
python -m pytest tests/test_interceptors/ -v
```

### With Coverage

```bash
python -m pytest tests/ --cov=vectorlens --cov-report=html
# Open htmlcov/index.html
```

---

## Known Limitations

### Attribution Granularity
- **Current**: Sentence-level detection
- **Future**: Token-level (v0.2+)
- Hallucinated phrase within a grounded sentence may not be flagged

### Perturbation Attribution Cost
- Requires **N additional LLM API calls** (N = chunks retrieved)
- Example: 10 chunks = 10 re-calls = 11× API cost
- Disabled by default, opt-in only

### Vector DB Support
- **Out-of-the-box**: OpenAI, Anthropic, Gemini, ChromaDB, Pinecone, FAISS, Weaviate, HuggingFace, LangChain, pgvector
- **Custom DBs**: Use manual event API (see Example 4)
- **Others** (Elasticsearch, Milvus): Contribute a patch or use manual API

### macOS + MPS (Metal Performance Shaders)
- Sentence-transformers with MPS can deadlock in multiprocessing
- **Workaround**: Preload model before serving:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
model.eval()

vectorlens.serve()
```

### Session Persistence
- Sessions stored **in-memory only**
- Server restart = session loss
- By design for local development
- SQLite backing planned

---

## Performance Notes

- **Embedding time**: ~50ms per sentence (CPU, all-MiniLM-L6-v2)
- **Attribution pipeline**: Runs in background, non-blocking
- **Dashboard**: WebSocket updates ~100ms latency
- **Memory**: ~300MB for model + session data (in-memory)

**For High-Throughput Systems**:
- Batch embedding calls together
- Sample (detect only 1-in-N requests)
- Disable perturbation attribution
- Consider streaming responses

---

## Extending VectorLens

### Adding a New LLM Provider

Create `vectorlens/interceptors/myprovider_patch.py`:

```python
from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import LLMRequestEvent, LLMResponseEvent
import functools

class MyProviderInterceptor(BaseInterceptor):
    def __init__(self):
        self._installed = False
        self._original_create = None

    def install(self):
        if self._installed:
            return
        try:
            import myprovider
            from myprovider import Client
        except ImportError:
            return

        self._original_create = Client.create
        Client.create = self._wrap_create(self._original_create)
        self._installed = True

    def uninstall(self):
        if not self._installed:
            return
        try:
            from myprovider import Client
            if self._original_create:
                Client.create = self._original_create
        except ImportError:
            pass
        self._installed = False

    def is_installed(self):
        return self._installed

    def _wrap_create(self, original):
        @functools.wraps(original)
        def wrapper(self_, **kwargs):
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])

            # Record request
            request_event = LLMRequestEvent(
                provider="myprovider",
                model=model,
                messages=messages,
            )
            bus.record_llm_request(request_event)

            # Call original
            start = time.time()
            response = original(self_, **kwargs)
            latency_ms = (time.time() - start) * 1000

            # Extract output and token counts
            output_text = extract_text_from_response(response)
            prompt_tokens = extract_prompt_tokens(response)
            completion_tokens = extract_completion_tokens(response)

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

Register in `vectorlens/interceptors/__init__.py`:

```python
from vectorlens.interceptors.myprovider_patch import MyProviderInterceptor

_INTERCEPTORS["myprovider"] = MyProviderInterceptor()
```

Test:
```bash
python -c "from vectorlens.interceptors import install_all; print(install_all())"
```

### Adding a New Vector DB

Same pattern as LLM providers. See `vectorlens/interceptors/chroma_patch.py` for reference.

Key differences:
- Patch a `query()` or `search()` method
- Extract `query_text`, `results` (list of chunks)
- Create `VectorQueryEvent` with `db_type`, `collection`, `results` list

### Changing the Detection Algorithm

Edit `vectorlens/detection/hallucination.py`. Keep the same interface:

```python
class HallucinationDetector:
    def detect(
        self,
        output_text: str,
        chunks: list[RetrievedChunk],
    ) -> list[OutputToken]:
        # Your implementation
        # Return list of OutputToken with is_hallucinated and chunk_attributions set
        pass
```

Tests in `tests/test_detection.py`.

---

## Troubleshooting

### "Port 7756 already in use"

```bash
lsof -i :7756
kill -9 <PID>
```

### Dashboard shows "No sessions"

Ensure `vectorlens.serve()` is called **before** your RAG code:

```python
import vectorlens
vectorlens.serve()  # FIRST

# Then your RAG code
results = collection.query(...)
```

### Slow detection (CPU at 100%)

First use loads the sentence-transformers model (~5s, one-time). Preload to avoid blocking:

```python
from vectorlens.detection.hallucination import _get_model
_get_model()  # Preload in main process

vectorlens.serve()
```

### "Sentence-transformers model not downloaded"

VectorLens auto-downloads on first use (100MB). Pre-download offline:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2')"
```

### macOS M1/M2 "Process freezes"

MPS + multiprocessing deadlock. Warm the model first:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
model.eval()  # Load to main process

vectorlens.serve()
```

### Tests fail with "No module named sentence_transformers"

Install with integration tests:

```bash
pip install "vectorlens[all]"
```

### "Interceptors not installed" warning

Some interceptors fail if their libraries aren't installed. This is expected. Install providers you need:

```bash
pip install "vectorlens[openai,chromadb]"
```

---

## Roadmap

- [x] v0.1: Sentence-level detection, 8 vector DBs, streaming support
- [ ] v0.2: Token-level detection, A/B comparison, pytest plugin
- [ ] v0.3: SQLite persistence, GitHub Actions plugin, multi-project views
- [ ] v1.0: MCP server, plugin system, multimodal RAG support

See [CHANGELOG.md](CHANGELOG.md) for version history and [CLAUDE.md](CLAUDE.md) for development notes.

---

## Contributing

Contributions welcome! For detailed development setup, see [CLAUDE.md](CLAUDE.md).

**Quick start**:

```bash
git clone https://github.com/Gustav-Proxi/vectorlens
pip install -e ".[all]"

# Run tests (fast)
python -m pytest tests/ -m "not integration"

# Build dashboard
cd dashboard && npm install && npm run build
cd .. && npm run dev  # Parallel with pytest
```

**Good first issues**: See [issues labeled "good first issue"](https://github.com/Gustav-Proxi/vectorlens/issues?q=label%3A%22good+first+issue%22).

---

## License

MIT License. See [LICENSE](LICENSE) for full text.

---

## Contact & Support

**Found a bug?** Open an [issue on GitHub](https://github.com/Gustav-Proxi/vectorlens/issues).

**Have a question?** Check [CLAUDE.md](CLAUDE.md) for architecture details and gotchas.

**Want to discuss?** Start a [discussion](https://github.com/Gustav-Proxi/vectorlens/discussions).
