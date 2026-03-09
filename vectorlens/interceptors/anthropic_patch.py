"""Interceptor for Anthropic API calls."""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import LLMRequestEvent, LLMResponseEvent

_logger = logging.getLogger(__name__)


class AnthropicInterceptor(BaseInterceptor):
    """Patches Anthropic API to record LLM requests and responses."""

    def __init__(self) -> None:
        self._installed = False
        self._install_lock = threading.Lock()
        self._original_create: Callable | None = None
        self._original_acreate: Callable | None = None

    def install(self) -> None:
        """Install Anthropic patches."""
        with self._install_lock:
            if self._installed:
                return

            try:
                from anthropic.resources.messages import Messages, AsyncMessages
            except ImportError:
                return

            # Patch sync create (anthropic.Anthropic client)
            self._original_create = Messages.create
            Messages.create = self._wrap_create(self._original_create)

            # Patch async create (anthropic.AsyncAnthropic client)
            self._original_acreate = AsyncMessages.create
            AsyncMessages.create = self._wrap_acreate(self._original_acreate)

            self._installed = True

    def uninstall(self) -> None:
        """Restore original Anthropic functions."""
        with self._install_lock:
            if not self._installed:
                return

            try:
                from anthropic.resources.messages import Messages, AsyncMessages
            except ImportError:
                return

            if self._original_create:
                Messages.create = self._original_create
            if self._original_acreate:
                AsyncMessages.create = self._original_acreate

            self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed

    def _wrap_create(self, original: Callable) -> Callable:
        """Wrap sync create method."""

        @functools.wraps(original)
        def wrapper(self_: Any, **kwargs: Any) -> Any:
            # Extract request parameters
            model = kwargs.get("model", "")
            messages = kwargs.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            # Create request event
            request_event = LLMRequestEvent(
                provider="anthropic",
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Link to recent vector query if available
            try:
                session = bus.get_or_create_session()
                if session.vector_queries:
                    last_query = session.vector_queries[-1]
                    elapsed = time.time() - last_query.timestamp
                    if elapsed < 5.0:
                        request_event.vector_query_id = last_query.id
            except Exception:
                _logger.debug("VectorLens: failed to link vector query", exc_info=True)

            try:
                bus.record_llm_request(request_event)
            except Exception:
                _logger.debug("VectorLens: failed to record request", exc_info=True)

            # Call original function
            start_time = time.time()
            response = original(self_, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, "content") and response.content:
                if isinstance(response.content, list) and len(response.content) > 0:
                    content_block = response.content[0]
                    if hasattr(content_block, "text"):
                        output_text = content_block.text

            if hasattr(response, "usage"):
                prompt_tokens = response.usage.input_tokens if hasattr(response.usage, "input_tokens") else 0
                completion_tokens = response.usage.output_tokens if hasattr(response.usage, "output_tokens") else 0

            # Calculate cost
            cost_usd = self._calculate_cost(model, prompt_tokens, completion_tokens)

            response_event = LLMResponseEvent(
                request_id=request_event.id,
                output_text=output_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            )

            try:
                bus.record_llm_response(response_event)
            except Exception:
                _logger.debug("VectorLens: failed to record response", exc_info=True)

            return response

        return wrapper

    def _wrap_acreate(self, original: Callable) -> Callable:
        """Wrap async create method."""

        @functools.wraps(original)
        async def wrapper(self_: Any, **kwargs: Any) -> Any:
            # Extract request parameters
            model = kwargs.get("model", "")
            messages = kwargs.get("messages") or []
            if not isinstance(messages, list):
                messages = []
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            # Create request event
            request_event = LLMRequestEvent(
                provider="anthropic",
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Link to recent vector query if available
            try:
                session = bus.get_or_create_session()
                if session.vector_queries:
                    last_query = session.vector_queries[-1]
                    elapsed = time.time() - last_query.timestamp
                    if elapsed < 5.0:
                        request_event.vector_query_id = last_query.id
            except Exception:
                _logger.debug("VectorLens: failed to link vector query", exc_info=True)

            try:
                bus.record_llm_request(request_event)
            except Exception:
                _logger.debug("VectorLens: failed to record request", exc_info=True)

            # Call original function
            start_time = time.time()
            response = await original(self_, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, "content") and response.content:
                if isinstance(response.content, list) and len(response.content) > 0:
                    content_block = response.content[0]
                    if hasattr(content_block, "text"):
                        output_text = content_block.text

            if hasattr(response, "usage"):
                prompt_tokens = response.usage.input_tokens if hasattr(response.usage, "input_tokens") else 0
                completion_tokens = response.usage.output_tokens if hasattr(response.usage, "output_tokens") else 0

            # Calculate cost
            cost_usd = self._calculate_cost(model, prompt_tokens, completion_tokens)

            response_event = LLMResponseEvent(
                request_id=request_event.id,
                output_text=output_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost_usd,
            )

            try:
                bus.record_llm_response(response_event)
            except Exception:
                _logger.debug("VectorLens: failed to record response", exc_info=True)

            return response

        return wrapper

    @staticmethod
    def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD based on token counts and model."""
        # Pricing as of knowledge cutoff (Feb 2025)
        # claude-3-5-sonnet: $3/1M input + $15/1M output
        # claude-3-haiku: $0.25/1M + $1.25/1M
        # Default fallback: $0.01/1K tokens

        if "claude-3-5-sonnet" in model or "claude-3.5-sonnet" in model:
            return (prompt_tokens * 3 / 1_000_000) + (completion_tokens * 15 / 1_000_000)
        elif "haiku" in model:
            return (prompt_tokens * 0.25 / 1_000_000) + (completion_tokens * 1.25 / 1_000_000)
        else:
            return (prompt_tokens + completion_tokens) * 0.01 / 1000
