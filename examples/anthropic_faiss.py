"""
Example 2: Anthropic Claude + FAISS
------------------------------------
Uses Anthropic's Claude as the LLM with FAISS for local vector search.
Good for privacy-sensitive use cases — no data leaves your machine except
the API call to Anthropic.

Install:
    pip install vectorlens[anthropic] faiss-cpu sentence-transformers numpy

Run:
    ANTHROPIC_API_KEY=sk-ant-... python examples/anthropic_faiss.py
"""

import os

import anthropic
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import vectorlens

# Start the local dashboard at http://127.0.0.1:7756
vectorlens.serve()

# ── Your RAG code — completely unchanged ──────────────────────────────────────

# Sample knowledge base
documents = [
    "Python was created by Guido van Rossum and first released in 1991.",
    "Python 3.0 was released in 2008 and is not backwards compatible with Python 2.",
    "Python is known for its readable syntax, using indentation to define code blocks.",
    "The Python Software Foundation manages Python's development and intellectual property.",
    "NumPy, Pandas, and PyTorch are popular Python libraries for data science and ML.",
]

# Build FAISS index
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(documents).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query
query = "Who created Python and when?"
query_vec = embed_model.encode([query]).astype("float32")
distances, indices = index.search(query_vec, k=3)
retrieved_chunks = [documents[i] for i in indices[0]]
context = "\n".join(retrieved_chunks)

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
message = client.messages.create(
    model="claude-3-5-haiku-20241022",
    max_tokens=512,
    messages=[
        {
            "role": "user",
            "content": f"Answer using only this context:\n{context}\n\nQuestion: {query}",
        }
    ],
)

print(message.content[0].text)
print("\n→ Open http://127.0.0.1:7756 to see attribution scores")

input("\nPress Enter to exit...")
