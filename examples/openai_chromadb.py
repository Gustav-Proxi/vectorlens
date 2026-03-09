"""
Example 1: OpenAI + ChromaDB
----------------------------
The most common RAG setup. VectorLens intercepts both the vector DB query
and the OpenAI completion — zero code changes to your existing pipeline.

Install:
    pip install vectorlens[openai] chromadb

Run:
    OPENAI_API_KEY=sk-... python examples/openai_chromadb.py
"""

import os

import chromadb
import openai

import vectorlens

# Start the local dashboard at http://127.0.0.1:7756
vectorlens.serve()

# ── Your RAG code — completely unchanged ──────────────────────────────────────

# Set up ChromaDB with some sample documents
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("docs")
collection.upsert(
    ids=["1", "2", "3"],
    documents=[
        "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "The Great Wall of China stretches over 13,000 miles and was built over many centuries.",
        "The Amazon rainforest covers approximately 5.5 million square kilometers in South America.",
    ],
)

# Run a query
query = "When was the Eiffel Tower built and where is it?"

results = collection.query(query_texts=[query], n_results=2)
retrieved_chunks = results["documents"][0]
context = "\n".join(retrieved_chunks)

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"Answer using only this context:\n{context}"},
        {"role": "user", "content": query},
    ],
)

print(response.choices[0].message.content)
print("\n→ Open http://127.0.0.1:7756 to see attribution scores")

# Keep server alive to explore the dashboard
input("\nPress Enter to exit...")
