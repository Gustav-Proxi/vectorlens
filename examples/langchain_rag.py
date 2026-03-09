"""
Example 3: LangChain LCEL RAG Pipeline
----------------------------------------
VectorLens intercepts at the httpx transport layer — it sees every LLM call
even inside LangChain chains, without any LangChain-specific code.

Install:
    pip install vectorlens[openai] langchain langchain-openai langchain-community chromadb

Run:
    OPENAI_API_KEY=sk-... python examples/langchain_rag.py
"""

import os

import chromadb
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import vectorlens

# Start the local dashboard at http://127.0.0.1:7756
vectorlens.serve()

# ── Your LangChain RAG code — completely unchanged ────────────────────────────

# Sample documents
docs = [
    Document(page_content="Quantum computing uses qubits instead of classical bits. Qubits can exist in superposition."),
    Document(page_content="Shor's algorithm can factor large integers exponentially faster than classical computers."),
    Document(page_content="Quantum entanglement allows qubits to be correlated regardless of physical distance."),
    Document(page_content="IBM, Google, and IonQ are leading companies in quantum hardware development."),
    Document(page_content="Quantum error correction is a major challenge — current qubits are noisy and prone to errors."),
]

# Build vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Build LCEL chain
prompt = ChatPromptTemplate.from_template(
    "Answer based only on this context:\n{context}\n\nQuestion: {question}"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the chain
question = "What makes quantum computers faster than classical ones?"
answer = chain.invoke(question)

print(answer)
print("\n→ Open http://127.0.0.1:7756 to see attribution scores")

input("\nPress Enter to exit...")
