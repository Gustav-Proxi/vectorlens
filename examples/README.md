# VectorLens Examples

Copy-pasteable RAG pipelines with VectorLens tracing enabled.

| File | Stack | What it shows |
|------|-------|---------------|
| `openai_chromadb.py` | OpenAI + ChromaDB | Most common RAG setup |
| `anthropic_faiss.py` | Anthropic + FAISS | Local vector search, privacy-focused |
| `langchain_rag.py` | LangChain LCEL + ChromaDB | Framework-level interception |

## Quick start

```bash
# Pick any example and run it — VectorLens opens at http://127.0.0.1:7756
OPENAI_API_KEY=sk-... python examples/openai_chromadb.py
```

Each script:
1. Starts the VectorLens dashboard
2. Runs a real RAG query
3. Pauses so you can explore attribution scores in the browser
