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

## Known Issues

None reported. If you discover a security issue, please report it per the instructions above.

## Future Improvements

- [ ] Optional SQLite session persistence (will be opt-in, on-disk)
- [ ] Configurable host/port binding
- [ ] Session expiration/cleanup
- [ ] Rate limiting (if exposing to network)

---

For other questions about VectorLens security, open a discussion on GitHub.
