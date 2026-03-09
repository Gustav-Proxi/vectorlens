"""Interceptor for OpenAI API calls."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import LLMRequestEvent, LLMResponseEvent


class OpenAIInterceptor(BaseInterceptor):
    """Patches OpenAI API to record LLM requests and responses."""

    def __init__(self) -> None:
        self._installed = False
        self._original_create: Callable | None = None
        self._original_acreate: Callable | None = None

    def install(self) -> None:
        """Install OpenAI patches."""
        if self._installed:
            return

        try:
            import openai
            from openai.resources.chat.completions import Completions
        except ImportError:
            return

        # Patch sync create
        self._original_create = Completions.create
        Completions.create = self._wrap_create(self._original_create)

        # Patch async create
        self._original_acreate = Completions.create_async
        Completions.create_async = self._wrap_acreate(self._original_acreate)

        self._installed = True

    def uninstall(self) -> None:
        """Restore original OpenAI functions."""
        if not self._installed:
            return

        try:
            from openai.resources.chat.completions import Completions
        except ImportError:
            return

        if self._original_create:
            Completions.create = self._original_create
        if self._original_acreate:
            Completions.create_async = self._original_acreate

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
            messages = kwargs.get("messages", [])
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            # Create request event
            request_event = LLMRequestEvent(
                provider="openai",
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Link to recent vector query if available
            session = bus.get_or_create_session()
            if session.vector_queries:
                last_query = session.vector_queries[-1]
                elapsed = time.time() - last_query.timestamp
                if elapsed < 5.0:
                    request_event.vector_query_id = last_query.id

            bus.record_llm_request(request_event)

            # Call original function
            start_time = time.time()
            try:
                response = original(self_, **kwargs)
            except Exception:
                raise

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    output_text = choice.message.content or ""

            if hasattr(response, "usage"):
                prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0
                completion_tokens = response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0

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

            bus.record_llm_response(response_event)

            return response

        return wrapper

    def _wrap_acreate(self, original: Callable) -> Callable:
        """Wrap async create method."""

        @functools.wraps(original)
        async def wrapper(self_: Any, **kwargs: Any) -> Any:
            # Extract request parameters
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            # Create request event
            request_event = LLMRequestEvent(
                provider="openai",
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Link to recent vector query if available
            session = bus.get_or_create_session()
            if session.vector_queries:
                last_query = session.vector_queries[-1]
                elapsed = time.time() - last_query.timestamp
                if elapsed < 5.0:
                    request_event.vector_query_id = last_query.id

            bus.record_llm_request(request_event)

            # Call original function
            start_time = time.time()
            try:
                response = await original(self_, **kwargs)
            except Exception:
                raise

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    output_text = choice.message.content or ""

            if hasattr(response, "usage"):
                prompt_tokens = response.usage.prompt_tokens if hasattr(response.usage, "prompt_tokens") else 0
                completion_tokens = response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0

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

            bus.record_llm_response(response_event)

            return response

        return wrapper

    @staticmethod
    def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD based on token counts and model."""
        # Pricing as of knowledge cutoff (Feb 2025)
        # gpt-4o: $5/1M input + $15/1M output
        # gpt-4o-mini: $0.15/1M + $0.60/1M
        # Default fallback: $0.01/1K tokens

        if "gpt-4o" in model:
            if "mini" in model:
                return (prompt_tokens * 0.15 / 1_000_000) + (completion_tokens * 0.60 / 1_000_000)
            else:
                return (prompt_tokens * 5 / 1_000_000) + (completion_tokens * 15 / 1_000_000)
        else:
            return (prompt_tokens + completion_tokens) * 0.01 / 1000
