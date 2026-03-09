"""Interceptor for Google Gemini API calls.

Supports both SDK versions:
- google-generativeai (legacy): google.generativeai.GenerativeModel.generate_content
- google-genai (new): google.genai.models.Models.generate_content
"""

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


class GeminiInterceptor(BaseInterceptor):
    """Patches Google Gemini APIs to record LLM requests and responses."""

    def __init__(self) -> None:
        self._installed = False
        self._install_lock = threading.Lock()
        self._original_generate_content_legacy: Callable | None = None
        self._original_generate_content_async_legacy: Callable | None = None
        self._original_generate_content_new: Callable | None = None
        self._original_generate_content_async_new: Callable | None = None

    def install(self) -> None:
        """Install Gemini patches for both SDK versions."""
        with self._install_lock:
            if self._installed:
                return

            # Try legacy SDK first (google-generativeai)
            self._install_legacy_sdk()

            # Try new SDK (google-genai)
            self._install_new_sdk()

            if (
                self._original_generate_content_legacy
                or self._original_generate_content_async_legacy
                or self._original_generate_content_new
                or self._original_generate_content_async_new
            ):
                self._installed = True

    def uninstall(self) -> None:
        """Restore original Gemini functions."""
        with self._install_lock:
            if not self._installed:
                return

            # Restore legacy SDK
            self._uninstall_legacy_sdk()

            # Restore new SDK
            self._uninstall_new_sdk()

            self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed

    def _install_legacy_sdk(self) -> None:
        """Install patches for google-generativeai (legacy SDK)."""
        try:
            import google.generativeai as genai

            # Patch sync generate_content
            self._original_generate_content_legacy = (
                genai.GenerativeModel.generate_content
            )
            genai.GenerativeModel.generate_content = self._wrap_generate_content_legacy(
                self._original_generate_content_legacy
            )

            # Patch async generate_content
            self._original_generate_content_async_legacy = (
                genai.GenerativeModel.generate_content_async
            )
            genai.GenerativeModel.generate_content_async = (
                self._wrap_generate_content_async_legacy(
                    self._original_generate_content_async_legacy
                )
            )
        except ImportError:
            pass

    def _uninstall_legacy_sdk(self) -> None:
        """Restore original functions for google-generativeai."""
        try:
            import google.generativeai as genai

            if self._original_generate_content_legacy:
                genai.GenerativeModel.generate_content = (
                    self._original_generate_content_legacy
                )
            if self._original_generate_content_async_legacy:
                genai.GenerativeModel.generate_content_async = (
                    self._original_generate_content_async_legacy
                )
        except ImportError:
            pass

    def _install_new_sdk(self) -> None:
        """Install patches for google-genai (new SDK)."""
        try:
            from google.genai.models import Models, AsyncModels

            # Patch sync generate_content
            self._original_generate_content_new = Models.generate_content
            Models.generate_content = self._wrap_generate_content_new(
                self._original_generate_content_new
            )

            # Patch async generate_content
            self._original_generate_content_async_new = AsyncModels.generate_content
            AsyncModels.generate_content = self._wrap_generate_content_async_new(
                self._original_generate_content_async_new
            )
        except ImportError:
            pass

    def _uninstall_new_sdk(self) -> None:
        """Restore original functions for google-genai."""
        try:
            from google.genai.models import Models, AsyncModels

            if self._original_generate_content_new:
                Models.generate_content = self._original_generate_content_new
            if self._original_generate_content_async_new:
                AsyncModels.generate_content = self._original_generate_content_async_new
        except ImportError:
            pass

    def _wrap_generate_content_legacy(self, original: Callable) -> Callable:
        """Wrap sync generate_content for legacy SDK."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract model name
            model = getattr(self_, "model_name", "unknown")

            # Extract contents (prompt) from args or kwargs
            contents = kwargs.get("contents", args[0] if args else "")

            # Create request event
            request_event = LLMRequestEvent(
                provider="gemini",
                model=model,
                messages=[{"role": "user", "content": str(contents)}],
                temperature=kwargs.get("generation_config", {}).get("temperature", 0.7)
                if isinstance(kwargs.get("generation_config"), dict)
                else 0.7,
                max_tokens=kwargs.get("generation_config", {}).get("max_output_tokens", 1024)
                if isinstance(kwargs.get("generation_config"), dict)
                else 1024,
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
            response = original(self_, *args, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Extract text from response
            if hasattr(response, "text"):
                output_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    if candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, "text"):
                            output_text = part.text

            # Extract token counts from usage_metadata
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    prompt_tokens = usage.prompt_token_count
                if hasattr(usage, "candidates_token_count"):
                    completion_tokens = usage.candidates_token_count

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

    def _wrap_generate_content_async_legacy(self, original: Callable) -> Callable:
        """Wrap async generate_content for legacy SDK."""

        @functools.wraps(original)
        async def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract model name
            model = getattr(self_, "model_name", "unknown")

            # Extract contents (prompt) from args or kwargs
            contents = kwargs.get("contents", args[0] if args else "")

            # Create request event
            request_event = LLMRequestEvent(
                provider="gemini",
                model=model,
                messages=[{"role": "user", "content": str(contents)}],
                temperature=kwargs.get("generation_config", {}).get("temperature", 0.7)
                if isinstance(kwargs.get("generation_config"), dict)
                else 0.7,
                max_tokens=kwargs.get("generation_config", {}).get("max_output_tokens", 1024)
                if isinstance(kwargs.get("generation_config"), dict)
                else 1024,
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
            response = await original(self_, *args, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Extract text from response
            if hasattr(response, "text"):
                output_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    if candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, "text"):
                            output_text = part.text

            # Extract token counts from usage_metadata
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    prompt_tokens = usage.prompt_token_count
                if hasattr(usage, "candidates_token_count"):
                    completion_tokens = usage.candidates_token_count

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

    def _wrap_generate_content_new(self, original: Callable) -> Callable:
        """Wrap sync generate_content for new SDK (google-genai)."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract model name
            model = kwargs.get("model", getattr(self_, "model", "unknown"))

            # Extract contents (prompt) from args or kwargs
            contents = kwargs.get("contents", args[0] if args else "")

            # Create request event
            request_event = LLMRequestEvent(
                provider="gemini",
                model=model,
                messages=[{"role": "user", "content": str(contents)}],
                temperature=kwargs.get("config", {}).get("temperature", 0.7)
                if isinstance(kwargs.get("config"), dict)
                else 0.7,
                max_tokens=kwargs.get("config", {}).get("max_output_tokens", 1024)
                if isinstance(kwargs.get("config"), dict)
                else 1024,
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
            response = original(self_, *args, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Extract text from response (new SDK response format)
            if hasattr(response, "text"):
                output_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    if candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, "text"):
                            output_text = part.text

            # Extract token counts (new SDK might have different field names)
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    prompt_tokens = usage.prompt_token_count
                if hasattr(usage, "candidates_token_count"):
                    completion_tokens = usage.candidates_token_count

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

    def _wrap_generate_content_async_new(self, original: Callable) -> Callable:
        """Wrap async generate_content for new SDK (google-genai)."""

        @functools.wraps(original)
        async def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract model name
            model = kwargs.get("model", getattr(self_, "model", "unknown"))

            # Extract contents (prompt) from args or kwargs
            contents = kwargs.get("contents", args[0] if args else "")

            # Create request event
            request_event = LLMRequestEvent(
                provider="gemini",
                model=model,
                messages=[{"role": "user", "content": str(contents)}],
                temperature=kwargs.get("config", {}).get("temperature", 0.7)
                if isinstance(kwargs.get("config"), dict)
                else 0.7,
                max_tokens=kwargs.get("config", {}).get("max_output_tokens", 1024)
                if isinstance(kwargs.get("config"), dict)
                else 1024,
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
            response = await original(self_, *args, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Extract text from response (new SDK response format)
            if hasattr(response, "text"):
                output_text = response.text
            elif hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    if candidate.content.parts:
                        part = candidate.content.parts[0]
                        if hasattr(part, "text"):
                            output_text = part.text

            # Extract token counts (new SDK might have different field names)
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if hasattr(usage, "prompt_token_count"):
                    prompt_tokens = usage.prompt_token_count
                if hasattr(usage, "candidates_token_count"):
                    completion_tokens = usage.candidates_token_count

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
        """Calculate cost in USD based on token counts and model.

        Pricing as of Feb 2025:
        - gemini-2.0-flash: $0.075/1M input + $0.30/1M output
        - gemini-1.5-pro: $1.25/1M + $5/1M
        - Default fallback: $0.001/1K tokens
        """
        if "2.0-flash" in model or "gemini-2-flash" in model:
            return (prompt_tokens * 0.075 / 1_000_000) + (
                completion_tokens * 0.30 / 1_000_000
            )
        elif "1.5-pro" in model or "gemini-1.5-pro" in model:
            return (prompt_tokens * 1.25 / 1_000_000) + (
                completion_tokens * 5 / 1_000_000
            )
        else:
            # Default fallback
            return (prompt_tokens + completion_tokens) * 0.001 / 1000
