"""Transport-layer interceptor via httpx.

Hooks httpx.AsyncClient.send + httpx.Client.send — the final send methods
used by OpenAI, Anthropic, and Gemini SDKs. SDK-version-agnostic.
"""

from __future__ import annotations

import functools
import json
import logging
import threading
import time
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import LLMRequestEvent, LLMResponseEvent

_logger = logging.getLogger(__name__)

# LLM API hosts to intercept
_LLM_HOSTS = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "gemini",
    "api.mistral.ai": "mistral",
}

# Generation endpoints to intercept
_LLM_PATHS = {"/v1/chat/completions", "/v1/messages", "/v1/generateContent"}


def _is_llm_request(request: Any) -> str | None:
    """Return provider name if this is an LLM API call, else None.

    Args:
        request: httpx.Request object

    Returns:
        Provider name ("openai", "anthropic", "gemini", etc.) or None
    """
    try:
        host = request.url.host
        provider = _LLM_HOSTS.get(host)
        if not provider:
            return None

        # Only intercept generation endpoints, not embeddings/models/etc
        path = request.url.path
        if any(path.endswith(p) for p in ("/chat/completions", "/messages", "/generateContent")):
            return provider
    except Exception:
        pass

    return None


async def _parse_request(request: Any, provider: str) -> LLMRequestEvent:
    """Parse httpx request to extract LLM call metadata.

    Args:
        request: httpx.Request object
        provider: Provider name ("openai", "anthropic", etc.)

    Returns:
        LLMRequestEvent with extracted metadata
    """
    model = ""
    messages: list[dict[str, Any]] = []
    system = ""
    temperature = 1.0
    max_tokens = 0

    try:
        # Parse request body
        content = request.content
        if not content:
            body: dict[str, Any] = {}
        else:
            body = json.loads(content)

        if provider == "openai":
            model = body.get("model", "")
            messages = body.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            system = body.get("system", "")
            temperature = body.get("temperature", 1.0)
            max_tokens = body.get("max_tokens", 0)

        elif provider == "anthropic":
            model = body.get("model", "")
            messages = body.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            system = body.get("system", "")
            temperature = body.get("temperature", 1.0)
            max_tokens = body.get("max_tokens", 0)

        elif provider == "gemini":
            model = body.get("model", "")
            contents = body.get("contents", [])
            messages = [
                {"role": c.get("role", "user"), "content": str(c.get("parts", ""))}
                for c in contents
                if isinstance(c, dict)
            ]
            gen_config = body.get("generationConfig", {})
            temperature = gen_config.get("temperature", 1.0)
            max_tokens = gen_config.get("maxOutputTokens", 0)

        else:
            model = body.get("model", "")

    except Exception:
        _logger.debug("httpx: failed to parse request body", exc_info=True)

    # Prepend system message if present
    if system and provider in ("openai", "anthropic"):
        messages = [{"role": "system", "content": system}] + messages

    event = LLMRequestEvent(
        provider=provider,
        model=model,
        messages=messages,
        system_prompt=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Link to recent vector query (within 5s)
    try:
        session = bus.get_or_create_session()
        if session.vector_queries:
            last = session.vector_queries[-1]
            if time.time() - last.timestamp < 5.0:
                event.vector_query_id = last.id
    except Exception:
        _logger.debug("httpx: failed to link vector query", exc_info=True)

    return event


def _parse_response(
    body: dict[str, Any], provider: str, request_event: LLMRequestEvent, latency_ms: float
) -> LLMResponseEvent:
    """Parse httpx response to extract LLM output and token counts.

    Args:
        body: Parsed JSON response body
        provider: Provider name ("openai", "anthropic", etc.)
        request_event: Corresponding request event
        latency_ms: Response latency in milliseconds

    Returns:
        LLMResponseEvent with extracted metadata
    """
    output_text = ""
    prompt_tokens = 0
    completion_tokens = 0

    try:
        if provider == "openai":
            choices = body.get("choices", [])
            if choices and isinstance(choices, list):
                output_text = choices[0].get("message", {}).get("content", "") or ""
            usage = body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else 0
            completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else 0

        elif provider == "anthropic":
            content = body.get("content", [])
            if content and isinstance(content, list) and len(content) > 0:
                output_text = content[0].get("text", "") or ""
            usage = body.get("usage", {})
            prompt_tokens = usage.get("input_tokens", 0) if isinstance(usage, dict) else 0
            completion_tokens = usage.get("output_tokens", 0) if isinstance(usage, dict) else 0

        elif provider == "gemini":
            candidates = body.get("candidates", [])
            if candidates and isinstance(candidates, list):
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts and isinstance(parts, list):
                    output_text = parts[0].get("text", "") or ""
            usage = body.get("usageMetadata", {})
            if isinstance(usage, dict):
                prompt_tokens = usage.get("promptTokenCount", 0)
                completion_tokens = usage.get("candidatesTokenCount", 0)

    except Exception:
        _logger.debug("httpx: failed to parse response body", exc_info=True)

    model = request_event.model
    cost = _calculate_cost(provider, model, prompt_tokens, completion_tokens)

    return LLMResponseEvent(
        request_id=request_event.id,
        output_text=output_text,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
    )


_COSTS = {  # (input_per_1M, output_per_1M)
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude-haiku-4": (0.25, 1.25),
    "gemini-2.0-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
}


def _calculate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD based on token counts and model.

    Args:
        provider: Provider name
        model: Model identifier
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    model_lower = model.lower()

    for key, (inp, out) in _COSTS.items():
        if key in model_lower:
            return (prompt_tokens * inp + completion_tokens * out) / 1_000_000

    # Fallback: $0.01 per 1K tokens
    return (prompt_tokens + completion_tokens) * 0.01 / 1000


def _make_async_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """Create async wrapper for httpx.AsyncClient.send.

    Args:
        original: Original send method

    Returns:
        Wrapped send method
    """

    @functools.wraps(original)
    async def wrapper(self: Any, request: Any, **kwargs: Any) -> Any:
        provider = _is_llm_request(request)

        if not provider:
            return await original(self, request, **kwargs)

        # Parse and record request
        request_event = None
        try:
            request_event = await _parse_request(request, provider)
            bus.record_llm_request(request_event)
        except Exception:
            _logger.debug("httpx: failed to record LLM request", exc_info=True)

        # Call original send
        start = time.time()
        response = await original(self, request, **kwargs)
        latency_ms = (time.time() - start) * 1000

        # Parse and record response
        if request_event:
            try:
                # Only process non-streaming responses
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    body = json.loads(response.content)
                    resp_event = _parse_response(body, provider, request_event, latency_ms)
                    bus.record_llm_response(resp_event)
            except Exception:
                _logger.debug("httpx: failed to record LLM response", exc_info=True)

        return response

    return wrapper


def _make_sync_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """Create sync wrapper for httpx.Client.send.

    Args:
        original: Original send method

    Returns:
        Wrapped send method
    """

    @functools.wraps(original)
    def wrapper(self: Any, request: Any, **kwargs: Any) -> Any:
        provider = _is_llm_request(request)

        if not provider:
            return original(self, request, **kwargs)

        # Parse and record request
        request_event = None
        try:
            model = ""
            messages: list[dict[str, Any]] = []
            system = ""
            temperature = 1.0
            max_tokens = 0

            # Parse request body synchronously
            try:
                content = request.content
                if content:
                    body: dict[str, Any] = json.loads(content)

                    if provider == "openai":
                        model = body.get("model", "")
                        messages = body.get("messages") or []
                        if not isinstance(messages, list):
                            messages = []
                        system = body.get("system", "")
                        temperature = body.get("temperature", 1.0)
                        max_tokens = body.get("max_tokens", 0)

                    elif provider == "anthropic":
                        model = body.get("model", "")
                        messages = body.get("messages") or []
                        if not isinstance(messages, list):
                            messages = []
                        system = body.get("system", "")
                        temperature = body.get("temperature", 1.0)
                        max_tokens = body.get("max_tokens", 0)

                    elif provider == "gemini":
                        model = body.get("model", "")
                        contents = body.get("contents", [])
                        messages = [
                            {"role": c.get("role", "user"), "content": str(c.get("parts", ""))}
                            for c in contents
                            if isinstance(c, dict)
                        ]
                        gen_config = body.get("generationConfig", {})
                        temperature = gen_config.get("temperature", 1.0)
                        max_tokens = gen_config.get("maxOutputTokens", 0)

                    else:
                        model = body.get("model", "")

                    # Prepend system message if present
                    if system and provider in ("openai", "anthropic"):
                        messages = [{"role": "system", "content": system}] + messages

            except Exception:
                _logger.debug("httpx: failed to parse sync request body", exc_info=True)

            request_event = LLMRequestEvent(
                provider=provider,
                model=model,
                messages=messages,
                system_prompt=system,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Link to recent vector query (within 5s)
            try:
                session = bus.get_or_create_session()
                if session.vector_queries:
                    last = session.vector_queries[-1]
                    if time.time() - last.timestamp < 5.0:
                        request_event.vector_query_id = last.id
            except Exception:
                _logger.debug("httpx: failed to link vector query", exc_info=True)

            bus.record_llm_request(request_event)
        except Exception:
            _logger.debug("httpx: failed to record sync LLM request", exc_info=True)

        # Call original send
        start = time.time()
        response = original(self, request, **kwargs)
        latency_ms = (time.time() - start) * 1000

        # Parse and record response
        if request_event:
            try:
                # Only process non-streaming responses
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" not in content_type:
                    body = json.loads(response.content)
                    resp_event = _parse_response(body, provider, request_event, latency_ms)
                    bus.record_llm_response(resp_event)
            except Exception:
                _logger.debug("httpx: failed to record sync LLM response", exc_info=True)

        return response

    return wrapper


class HttpxTransportInterceptor(BaseInterceptor):
    """Patches httpx.AsyncClient.send and httpx.Client.send.

    SDK-agnostic: works with any version of OpenAI, Anthropic, Gemini SDKs
    since they all use httpx for HTTP. Intercepts at the transport layer,
    not at SDK method level — immune to SDK API changes.
    """

    def __init__(self) -> None:
        self._installed = False
        self._install_lock = threading.Lock()
        self._original_async_send: Callable[..., Any] | None = None
        self._original_sync_send: Callable[..., Any] | None = None

    def install(self) -> None:
        """Install httpx transport patches."""
        with self._install_lock:
            if self._installed:
                return

            try:
                import httpx
            except ImportError:
                return

            self._original_async_send = httpx.AsyncClient.send
            httpx.AsyncClient.send = _make_async_wrapper(self._original_async_send)

            self._original_sync_send = httpx.Client.send
            httpx.Client.send = _make_sync_wrapper(self._original_sync_send)

            self._installed = True

    def uninstall(self) -> None:
        """Restore original httpx functions."""
        with self._install_lock:
            if not self._installed:
                return

            try:
                import httpx

                if self._original_async_send:
                    httpx.AsyncClient.send = self._original_async_send
                if self._original_sync_send:
                    httpx.Client.send = self._original_sync_send
            except ImportError:
                pass

            self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed
