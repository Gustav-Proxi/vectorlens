# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | ✅ (Active support) |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

If you discover a security issue, please email the maintainers with the subject line `VectorLens Security Vulnerability`.

**What to include:**
- A clear description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact
- Suggested fix (if you have one)

**Response time:** We aim to respond within 48 hours and will work with you on a fix and disclosure timeline.

## Security Scope

VectorLens is a **local development debugging tool**, not production infrastructure:

- **Default binding**: 127.0.0.1 (localhost only)
- **Use case**: Running on your machine or behind corporate firewall
- **Data handling**: All data stays local; no network calls to external services
- **Authentication**: Not implemented (not needed for localhost)

## Security Best Practices

### 1. Do Not Expose the Dashboard to the Internet

VectorLens is designed for local development only. **Never expose port 7756 to the public internet.**

```python
# Good: localhost only (default)
vectorlens.serve(host="127.0.0.1", port=7756)

# Bad: exposed to network
vectorlens.serve(host="0.0.0.0", port=7756)
```

### 2. LLM API Keys

VectorLens intercepts LLM requests but **does not store API keys**. Your API keys remain in memory only for the duration of the request:

- Keys are not logged
- Keys are not transmitted externally
- Keys are not stored to disk

However, when running `vectorlens.serve()` during development, ensure your API keys are only set in environment variables or secure credential stores.

```python
# Good: use environment variable
import os
api_key = os.getenv("OPENAI_API_KEY")

# Avoid: hardcoding keys
api_key = "sk-..."  # Never do this!
```

### 3. Session Data

Session data (LLM prompts, retrieved chunks) is stored **in-memory only** by default:

- Sessions are lost when the server restarts
- No persistent storage to disk
- Not synced to cloud services

For production use cases, implement your own persistence layer.

### 4. Dependencies

VectorLens vendors only essential dependencies:

- `fastapi`, `uvicorn` — web framework
- `sentence-transformers` — ML model (open-source)
- `numpy`, `aiosqlite`, `httpx` — utilities

Check `pyproject.toml` for the full list. All are widely-used, well-maintained packages.

To audit dependencies:

```bash
pip install pip-audit
pip-audit
```

## Advanced Threat Model & Optimization Limits

Following an exhaustive "Red Team" deep sweep (March 2026), the following theoretical edge cases, vulnerabilities, and optimization limits were identified. **These have deliberately not been patched** to maintain the project's minimal local footprint, but developers should be aware of them:

### 1. `httpx` Streaming Context Manager Crash (Resilience / DoS)
**Vector**: The `httpx_transport.py` interceptor wraps streaming responses in `_StreamingResponseWrapper` using `__getattr__`. Python does not forward magic dunder methods (like `__aenter__` or `__enter__`) through `__getattr__`.
**Impact**: If a host application uses `async with client.send(..., stream=True)` with the interceptor active, the host app will immediately crash with a `TypeError`.

### 2. Unbounded JSON Deserialization in SSE Streams (DoS)
**Vector**: In the `httpx` streaming parser, every Server-Sent Event (SSE) line is parsed via `json.loads(data)` without length limits.
**Impact**: If a malicious or hijacked LLM endpoint returns a single chunk containing a 500MB JSON payload, it will block the Python event loop and cause an Out-Of-Memory (OOM) crash in the host application.

### 3. Weak Cross-Site WebSocket Hijacking (CSWSH) Enforcement
**Vector**: The WebSocket route in `app.py` checks `if origin and origin not in _ALLOWED_ORIGINS`. If the `Origin` header is missing entirely or empty, the condition `if origin` evaluates to false, bypassing the whitelist check.
**Impact**: While standard web browsers *always* send the `Origin` header for WebSockets, a specially crafted non-browser attack script on the local network could bypass this check. Strict enforcement (`if not origin or origin not in...`) is recommended if exposing to a network.

### 4. SentenceTransformer OOM via Giant Prompts (Optimization)
**Vector**: Both `perturbation.py` and `hallucination.py` pass the raw `original_output` directly to `SentenceTransformer.encode()`. The token limits are managed internally by the HuggingFace backend, but the pre-tokenization string is loaded into memory simultaneously.
**Impact**: If an LLM returns a 100,000-token output, VectorLens passes this massive string to the C++ ML backend. This can cause massive RAM spikes or thread-blocking in the attribution pipeline before it successfully truncates the input.

### 5. Unbounded Session Memory Bloat (Optimization)
**Vector**: `SessionBus` retains the last 200 sessions in volatile RAM. A "Session" includes all raw input messages, retrieved chunks, and model outputs.
**Impact**: 200 sessions × 100k context tokens × 4 bytes = ~80MB per chunk. A heavy debugging day could silently consume gigabytes of the user's RAM. Persistent bounded SQLite storage is recommended.

## Known Issues

None reported. If you discover a security issue, please report it per the instructions above.

## Future Improvements

- [ ] Optional SQLite session persistence (will be opt-in, on-disk)
- [ ] Configurable host/port binding
- [ ] Session expiration/cleanup
- [ ] Rate limiting (if exposing to network)

---

For other questions about VectorLens security, open a discussion on GitHub.
