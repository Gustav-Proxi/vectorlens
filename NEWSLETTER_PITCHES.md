# VectorLens Newsletter Pitches

## 1. PyCoder's Weekly

**Subject:** RAG hallucination debugging just got zero-config

**Pitch:**

Most RAG systems hallucinate, and most teams have no idea why. Is it bad retrieval? The model making things up? VectorLens answers that in 2 lines of code: `import vectorlens; vectorlens.serve()`. Local dashboard shows token-level attribution—which exact retrieved chunks caused each hallucinated sentence. No cloud, no signup, pure Python. MIT licensed, supports OpenAI/Anthropic/Gemini/Mistral + ChromaDB/Pinecone/FAISS. Check it out: https://github.com/Gustav-Proxi/vectorlens

---

## 2. Python Bytes

**Subject:** One-liner RAG debugger that actually works

**Pitch:**

Been wrestling with RAG hallucinations? Same. VectorLens is what we've all been missing—drop it into any Python RAG pipeline and get a local dashboard showing which retrieved chunks caused which hallucinations. No API calls, no cloud account, just `pip install vectorlens` and `vectorlens.serve()`. Token-level attribution on OpenAI, Anthropic, Gemini, Mistral, LangChain, you name it. MIT license. GitHub: https://github.com/Gustav-Proxi/vectorlens

---

## 3. Console.dev

**Subject:** VectorLens: Token-level attribution for RAG hallucinations

**Pitch:**

Debugging RAG quality is broken. You get a hallucination, but no visibility into whether it's the retrieval layer, the LLM, or their interaction. VectorLens solves this: `import vectorlens; vectorlens.serve()` launches a dashboard with token-level attribution—exactly which chunks retrieved caused which hallucinated outputs. Works out-of-the-box with OpenAI, Anthropic, Gemini, Mistral, ChromaDB, Pinecone, FAISS, LangChain. Reduces debugging time from hours to minutes. MIT licensed. https://github.com/Gustav-Proxi/vectorlens

---

## 4. The Rundown AI

**Subject:** VectorLens: RAG hallucination debugger (zero-config, MIT)

**Pitch:**

RAG is broken until you can debug it. VectorLens: zero-config hallucination debugger. Two lines of Python and you get a local dashboard showing token-level attribution—which retrieved chunks caused each hallucinated sentence. Works with all major LLM APIs (OpenAI, Anthropic, Gemini, Mistral) and vector DBs (ChromaDB, Pinecone, FAISS, Weaviate). No cloud, no signup, MIT license. https://github.com/Gustav-Proxi/vectorlens

---

## 5. Ben's Bites

**Subject:** RAG hallucinations? VectorLens shows you exactly why

**Pitch:**

VectorLens is wild. Drop it into your RAG pipeline (`pip install vectorlens`, then `vectorlens.serve()`), get a dashboard showing which retrieved chunks caused which hallucinations—token level. Works with OpenAI, Anthropic, Gemini, Mistral, LangChain, ChromaDB, Pinecone, FAISS. Local, zero-config, MIT. Finally know if your retrieval sucks or your model does. https://github.com/Gustav-Proxi/vectorlens

---

## Pitch Notes

- **PyCoder's Weekly**: Professional tone, emphasizes ecosystem support (multiple LLM/DB providers), developer-first framing
- **Python Bytes**: Conversational, casual, relatable pain point ("we've all been missing this")
- **Console.dev**: Editorial focus on workflow impact, concrete time savings, minimal marketing language
- **The Rundown AI**: Direct, AI-native audience, emphasizes the speed/visibility gain for RAG workflows
- **Ben's Bites**: Casual, hype-friendly, short/snappy, personality-driven, link-heavy

All pitches:
- Lead with the problem (RAG hallucination debugging is broken)
- Include one technical hook (token-level attribution, local dashboard)
- Name-drop supported providers (builds credibility)
- Include GitHub link
- Avoid marketing copy ("powerful", "game-changing", etc.)
- Sound like a developer sharing something they built
