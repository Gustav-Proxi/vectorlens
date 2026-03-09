# VectorLens

**See why your RAG hallucinates — token-level attribution in 30 seconds, zero config.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

> **Dashboard screenshot**: Shows hallucinated sentence highlighted in red, retrieved chunks panel below with attribution scores sorted by influence.

## Why VectorLens

**The Problem**: RAG debugging is painful. When your LLM hallucinates, you're left guessing which retrieved chunk caused it. Print statements everywhere. Log files. Sift through traces. Repeat.

**What Exists Doesn't Solve This**:
- **RAGxplorer** visualizes RAG flows but doesn't detect hallucinations
- **Arize Phoenix** requires cloud signup and sends data externally
- **LangSmith** requires enterprise accounts
- **None** do token-level attribution without heavy instrumentation

**The Solution**: Three lines of code. Local dashboard. Instant attribution.

```python
import vectorlens
vectorlens.serve()  # Open http://127.0.0.1:7756

# Your RAG code — unchanged
results = collection.query(query_texts=["Q"], n_results=5)
response = client.chat.completions.create(model="gpt-4o", messages=[...])
# Dashboard now shows which chunks caused each hallucination
```

**What You Get**:
- Which sentences in the output are hallucinated (highlighted red)
- Which retrieved chunks caused each hallucination
- Attribution scores (0–100%) showing each chunk's influence
- Overall groundedness percentage
- All running locally, no signup, no data leaving your machine

## How It Works

```
Your RAG code (unchanged)
    │
    ▼
[Interceptors]──────────────────────────────────────────────┐
 OpenAI │ Anthropic │ Gemini │ ChromaDB │ Pinecone │ FAISS  │
    │                                                        │
    ▼                                                        │
[Session Bus] ← vector queries, LLM requests/responses      │
    │                                                        │
    ▼                                                        │
[Attribution Pipeline]                                       │
 1. sentence-transformers embeds output + chunks             │
 2. cosine similarity → hallucination detection              │
 3. perturbation: drop chunk → re-run → measure divergence   │
    │                                                        │
    ▼                                                        │
[FastAPI + WebSocket] → [React Dashboard]                   ◄┘
```

**The Flow**:

1. **Interception** (transparent): Monkey-patches LLM client methods and vector DB query methods. Zero changes to your code.

2. **Event Bus** (in-process): Every vector query, LLM request/response captured in a thread-safe event bus. No network calls, no external services.

3. **Attribution Pipeline** (background thread): Runs hallucination detection and chunk attribution in parallel. Never blocks your main code.

4. **Dashboard** (local): React UI shows real-time attribution updates via WebSocket.

**Key Details**:
- Uses `sentence-transformers/all-MiniLM-L6-v2` (22MB, CPU-only) to embed sentences and chunks
- Cosine similarity threshold 0.4 to detect hallucinations (conservative, fewer false positives)
- Top-3 chunks per sentence for attribution (to avoid noise)
- Attribution scores computed as normalized similarity weights

## Installation

```bash
# Base install
pip install vectorlens

# With your LLM provider (choose one or more)
pip install "vectorlens[openai]"
pip install "vectorlens[anthropic]"
pip install "vectorlens[gemini]"

# With your vector DB
pip install "vectorlens[chromadb]"
pip install "vectorlens[pinecone]"

# All at once
pip install "vectorlens[all]"
```

**Requirements**: Python 3.11+. Minimal dependencies (FastAPI, uvicorn, sentence-transformers, numpy, aiosqlite, httpx).

## Quickstart

### Example 1: OpenAI + ChromaDB (Most Common)

```python
import vectorlens

# Start the dashboard
vectorlens.serve()  # Opens http://127.0.0.1:7756 in browser

# Your existing RAG code — zero changes
import chromadb
import openai

collection = chromadb.Client().get_or_create_collection("docs")
client = openai.OpenAI()

# Load documents
collection.add(
    ids=["id1", "id2", "id3"],
    documents=[
        "Attention is a mechanism that allows models to focus on relevant parts of input.",
        "The transformer architecture uses self-attention to compute relationships between tokens.",
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

# Call LLM
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
# - Retrieved chunks and their vector similarity scores
# - Which output sentences are hallucinated
# - Attribution: which chunks explain which sentences
# - Groundedness: % of output grounded in retrieved chunks
```

Visit http://127.0.0.1:7756 in your browser while the script runs.

### Example 2: Anthropic + Custom DB (Manual Event API)

For unsupported vector DBs, manually record events:

```python
import vectorlens
from vectorlens.session_bus import bus
from vectorlens.types import VectorQueryEvent, RetrievedChunk

vectorlens.serve()

# Your custom vector DB search
async def my_pgvector_search(query: str) -> list[dict]:
    """Example: PostgreSQL + pgvector"""
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

# Use in your RAG
from anthropic import Anthropic

async def rag_query(question: str):
    chunks = await my_pgvector_search(question)
    context = "\n".join([c["text"] for c in chunks])

    client = Anthropic()
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.content[0].text

# Run it
result = asyncio.run(rag_query("What is quantum computing?"))
# Dashboard shows attribution to pgvector results
```

## Supported Integrations

| Provider | Captures | Support |
|----------|----------|---------|
| **OpenAI** | model, tokens, cost, latency | ✓ Native |
| **Anthropic** | model, tokens, cost, latency | ✓ Native |
| **Google Gemini** | model (SDK v1 & v2), tokens, latency | ✓ Native |
| **LangChain** | LLM calls, retrievers, chain steps | ✓ Native |
| **ChromaDB** | similarity scores, metadata, collection | ✓ Native |
| **Pinecone** | similarity scores, namespace, metadata | ✓ Native |
| **FAISS** | L2/dot-product distances, metadata | ✓ Native |
| **Weaviate** | similarity scores, metadata, class | ✓ Native |
| **pgvector** | SQL similarity operators, metadata | ✓ Native |
| **HuggingFace Transformers** | model, tokens, latency | ✓ Native |
| **Custom DB (Elasticsearch, Milvus, etc.)** | via manual event API | ✓ See Example 2 |

**Cost Calculation** (as of Feb 2025):
- GPT-4o: $5/1M input tokens + $15/1M output tokens
- Claude 3.5 Sonnet: $3/1M input + $15/1M output
- Gemini 2.0 Flash: $0.075/1M input + $0.30/1M output

## Dashboard Features

### Session Sidebar (Left Panel)
- **Live Session** (green dot): Current active session with real-time event updates
- **Session History** (📋 icon): Previously recorded sessions (cached in localStorage, persist across server restarts)
- Click any session to view its attribution results
- Sessions auto-create on first event

### Groundedness Meter (Top Center)
- **0–100%** scale: Percentage of output grounded in retrieved chunks
- **Color coding**:
  - Red (0–30%): Mostly hallucinated — highly skeptical
  - Yellow (30–70%): Mixed — verify claims carefully
  - Green (70–100%): Mostly grounded — likely reliable

### LLM Output Panel (Center)
- Full response text with sentence-level highlighting
  - **Red background**: Hallucinated sentence (cosine similarity to chunks < 0.4)
  - **Normal text**: Grounded in retrieved chunks
- **Hover over any sentence** to see contributing chunks with attribution percentages
- Metadata shown: token counts, latency (ms), estimated cost (USD)

### Retrieved Chunks Panel (Right)
- Sorted by attribution score (descending)
- Each chunk shows:
  - **Similarity**: Vector DB's original similarity score
  - **Attribution %**: How much this chunk contributed to the output
  - **Metadata**: Custom fields from vector DB (source, page number, etc.)
  - **⚠️ Badge**: If chunk contributed to hallucination
- Click to highlight related sentences in output

## Streaming Support

VectorLens now captures streaming responses (`stream=True`). SSE chunks are intercepted at the httpx transport layer, and the full text is reconstructed after the stream completes. Works with:
- OpenAI (streaming)
- Anthropic (streaming)
- Google Gemini (streaming)
- Mistral (streaming)

Note: Streaming responses don't include exact token counts in SSE chunks; VectorLens estimates completion tokens from word count.

## Multi-turn Agents

VectorLens tracks multi-turn conversation DAGs via `parent_request_id` linking. For multi-turn agents:
- Call `bus.start_conversation()` to get a `conversation_id` (groups related sessions)
- Each `LLMRequestEvent` includes `parent_request_id` linking child calls to parent
- The `chain_step` field labels the role (e.g., "agent", "tool_use")
- Dashboard shows connected call tree instead of isolated calls

Example:
```python
vectorlens.serve()

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent

# LangChain agent automatically linked via parent_request_id
agent = create_openai_functions_agent(llm=ChatOpenAI(), ...)
executor = AgentExecutor(agent=agent, tools=tools)

# Multi-turn calls automatically part of same conversation DAG
result = executor.invoke({"input": "What is the capital of France?"})
# Dashboard shows: user query → agent call → tool call → agent response
```

## API Reference

### `vectorlens.serve()`

Start the dashboard server in a background thread and install interceptors.

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
- Sets up auto-attribution pipeline
- If already running, logs warning and returns current URL

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
- Logs completion

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

**Behavior**:
- If `session_id` provided: returns `http://{host}:{port}/sessions/{session_id}`
- If `session_id` is None: returns base URL `http://{host}:{port}`

**Example**:
```python
sess_id = vectorlens.new_session()
url = vectorlens.get_session_url(sess_id)
print(f"View this session: {url}")
```

## How Attribution Works

### Hallucination Detection (Sentence-Level)

VectorLens detects hallucinations by comparing semantic similarity:

**Algorithm**:
1. Split LLM output into sentences using regex (split on `. ` and final `.`)
2. Embed each sentence using `all-MiniLM-L6-v2` (384-dimensional vectors)
3. Embed each retrieved chunk using the same model
4. Compute cosine similarity between each sentence and all chunks
5. **Detect as hallucinated** if `max(similarities) < 0.4`
6. For grounded sentences, record top-3 most similar chunks as attribution weights

**Threshold Justification**: 0.4 cosine similarity is conservative:
- Avoids false positives (marking correct sentences as hallucinated)
- Works well with sentence-transformers' semantic space
- Tuned empirically on common RAG failure cases

**Conditional Attribution**: Deep attribution (perturbation or attention rollout) only runs when hallucinations detected. Grounded responses skip expensive analysis (~50ms vs ~500ms savings).

**Attribution Methods**:
- **LIME perturbation** (API models): K=7 random binary masks over chunks. Ridge regression fits mask vectors to output similarity scores. Cost: exactly 7 LLM calls regardless of chunk count.
- **Attention rollout** (local HuggingFace models): Extracts token-level attention weights from model internals. Zero extra LLM calls.

**Example**:
```
Output: "Transformers use self-attention to compute token relationships."
Chunk 1: "The transformer architecture uses self-attention to compute relationships..."
         Similarity: 0.85 (GROUNDED)

Chunk 2: "BERT was trained on Wikipedia data."
         Similarity: 0.15 (NOT RELEVANT)

Max similarity: 0.85 > 0.4 → Grounded
```

**Limitations**:
- Sentence-level, not token-level (MVP)
- Hallucinated phrase within a grounded sentence may not be flagged
- May miss subtle semantic errors (e.g., swapped facts)

### Perturbation Attribution (Optional, Expensive)

Measure chunk importance by dropping each and measuring output change:

```python
from vectorlens.attribution.perturbation import PerturbationAttributor

async def my_llm_call(messages: list[dict]) -> str:
    """Your LLM call function"""
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    return response.choices[0].message.content

attributor = PerturbationAttributor(llm_caller=my_llm_call)

# Compute attribution via perturbation
chunks_with_scores = await attributor.compute(
    original_messages=messages,
    chunks=retrieved_chunks,
    original_output=original_response_text,
    output_tokens=output_tokens
)

# For each chunk:
# attribution_score = 1 - cosine_similarity(original_output, output_without_chunk)
# Higher score = dropping chunk changed output more = chunk was important
```

**Cost**: N additional LLM API calls (N = number of chunks). If you retrieve 10 chunks, 10 re-calls. Expensive but accurate.

**Disabled by default** — not called automatically. Opt-in only via explicit code.

## Architecture

```
vectorlens/
├── __init__.py
│   └── Public API: serve(), stop(), new_session(), get_session_url()
│
├── types.py
│   └── Shared dataclasses (EventType, Session, LLMRequestEvent, etc.)
│
├── session_bus.py
│   └── Thread-safe in-process event bus. Interceptors publish, pipeline subscribes.
│
├── pipeline.py
│   └── Auto-attribution pipeline. Subscribes to llm_response events.
│       Runs detection + attribution in background thread.
│
├── interceptors/
│   ├── __init__.py
│   │   └── Registry: install_all(), uninstall_all(), get_installed()
│   │
│   ├── base.py
│   │   └── BaseInterceptor abstract class
│   │
│   ├── httpx_transport.py
│   │   └── Patches httpx.AsyncClient.send / httpx.Client.send (SDK-agnostic)
│   │
│   ├── openai_patch.py
│   │   └── Patches openai.resources.chat.completions.Completions.create (fallback)
│   │
│   ├── anthropic_patch.py
│   │   └── Patches anthropic.resources.messages.Messages.create (fallback)
│   │
│   ├── gemini_patch.py
│   │   └── Patches google.generativeai.GenerativeModel.generate_content (fallback)
│   │
│   ├── langchain_patch.py
│   │   └── Patches langchain.BaseChatModel + BaseRetriever
│   │
│   ├── chroma_patch.py
│   │   └── Patches chromadb.api.models.Collection.query
│   │
│   ├── pinecone_patch.py
│   │   └── Patches pinecone.Index.query
│   │
│   ├── faiss_patch.py
│   │   └── Wraps faiss.Index.search (numpy array wrapper)
│   │
│   ├── weaviate_patch.py
│   │   └── Patches weaviate.Client query methods
│   │
│   ├── pgvector_patch.py
│   │   └── Patches SQLAlchemy AsyncSession.execute / Session.execute
│   │
│   └── transformers_patch.py
│       └── Patches huggingface pipeline inference
│
├── detection/
│   └── hallucination.py
│       ├── HallucinationDetector class
│       ├── _get_model() singleton (lazy-loads sentence-transformers)
│       ├── _split_sentences() regex-based sentence splitting
│       └── detect() returns list[OutputToken] with hallucination flags
│
├── attribution/
│   ├── perturbation.py
│   │   ├── PerturbationAttributor class
│   │   ├── compute() async method (N+1 perturbation)
│   │   ├── compute_lime() async method (K=7 fixed cost)
│   │   └── _perturb_chunk() async method
│   │
│   └── attention.py
│       ├── AttentionAttributor class
│       └── compute() for local HuggingFace models
│
└── server/
    ├── app.py
    │   └── FastAPI app definition + static file serving
    │
    ├── api.py
    │   ├── REST endpoints
    │   │   ├── GET /status → server status + installed interceptors
    │   │   ├── GET /sessions → list all sessions
    │   │   ├── GET /sessions/{id} → full session details
    │   │   ├── GET /sessions/{id}/attributions → attributions only
    │   │   ├── POST /sessions/new → create session
    │   │   └── DELETE /sessions/{id} → delete session
    │   │
    │   └── Pydantic models for serialization
    │
    └── static/
        └── Built React dashboard (dist files from npm build)

dashboard/                # React + TypeScript + Tailwind
├── src/
│   ├── components/      # UI components (SessionList, Output, Chunks, etc.)
│   ├── hooks/           # Custom hooks (useWebSocket, useSessions, etc.)
│   ├── App.tsx          # Main app component
│   └── index.css        # Tailwind styles
├── public/
├── package.json
└── tsconfig.json
```

**Key Design Patterns**:

1. **Monkey-patching (interceptors)**: Directly patches client library methods. Zero wrapper overhead, zero changes to user code.

2. **Event bus pattern**: SessionBus publishes events. Pipeline subscribes. Loose coupling — easy to add new consumers (e.g., logging, analytics).

3. **Non-blocking attribution**: Background daemon thread. Main code never waits. User sees results instantly.

4. **Lazy-loaded models**: Sentence-transformers model loads on first hallucination detection, not on import. No upfront cost if detection never runs.

5. **Thread-safe sessions**: All state in SessionBus with locks. Safe for multithreaded usage.

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
# Open htmlcov/index.html to view
```

## Known Limitations

### Attribution Granularity
- Current: **sentence-level** detection
- Future: **token-level** (requires different model)
- A hallucinated phrase within a grounded sentence may not be flagged

### Perturbation Attribution Cost
- Requires **N additional LLM API calls** (N = chunks retrieved)
- Example: 10 chunks = 10 re-calls = 11× API cost
- Disabled by default
- Opt-in only for expensive analyses

### Vector DB Support
- Out-of-the-box: OpenAI, Anthropic, Gemini, ChromaDB, Pinecone, FAISS, Weaviate, HuggingFace, LangChain, pgvector
- Custom DBs: Use manual event API (see Example 2)
- Elasticsearch, Milvus, others: Contribute a patch or use manual API

### macOS + MPS (Metal Performance Shaders)
- Sentence-transformers with MPS can deadlock in multiprocessing
- **Workaround**:
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer("all-MiniLM-L6-v2")
  model.eval()  # Preload in main process

  vectorlens.serve()
  ```

### Session Persistence
- Sessions stored **in-memory only**
- Server restart = session loss
- By design for local development
- SQLite backing planned (TODO)

### Port 7756
- Server hard-binds to port 7756
- No automatic fallback
- Future: make configurable

## Extending VectorLens

### Adding a New LLM Provider

1. Create `vectorlens/interceptors/myprovider_patch.py`:

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
            return  # Library not installed, skip silently

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
            # Extract parameters
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

2. Register in `vectorlens/interceptors/__init__.py`:

```python
from vectorlens.interceptors.myprovider_patch import MyProviderInterceptor

_INTERCEPTORS["myprovider"] = MyProviderInterceptor()
```

3. Test:
```bash
python -c "from vectorlens.interceptors import install_all; print(install_all())"
```

### Adding a New Vector DB

Same pattern. See `vectorlens/interceptors/chroma_patch.py` for reference.

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

## Performance Notes

- **Embedding time**: ~50ms per sentence (CPU, all-MiniLM-L6-v2)
- **Attribution pipeline**: Runs in background, non-blocking
- **Dashboard**: WebSocket updates ~100ms latency
- **Memory**: ~300MB for model + session data (in-memory)

**For High-Throughput Systems**:
- Batch embedding calls together
- Sample (detect only 1-in-N requests)
- Disable perturbation attribution
- Consider streaming responses (now supported)

## License

MIT License. See LICENSE file for full text.

## Contributing

Contributions welcome!

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Add tests for new functionality
4. Run `pytest tests/ -m "not integration"` before committing
5. Follow black + isort formatting (auto-applied via pre-commit hooks)
6. Submit a pull request

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

### "interceptors not installed" warning

Some interceptors fail if their libraries aren't installed. This is expected. Install providers you need:

```bash
pip install "vectorlens[openai,chromadb]"
```

## Roadmap

- [ ] Token-level (not sentence-level) hallucination detection
- [ ] Attention-based attribution (no extra LLM calls)
- [ ] SQLite session persistence
- [ ] Custom similarity threshold configuration
- [ ] WebSocket compression
- [ ] Python 3.10 support
- [ ] Multimodal RAG (images, audio)

## Contact & Support

**Found a bug?** Open an issue on GitHub.

**Have a question?** Check CLAUDE.md for developer notes and architecture details.

**Want to discuss?** Start a discussion in GitHub Issues.
