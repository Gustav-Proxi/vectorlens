"""Interceptor for HuggingFace transformers library calls.

Patches transformers.Pipeline and transformers.GenerationMixin to record
LLM requests and responses.
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, Tuple, Optional

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import LLMRequestEvent, LLMResponseEvent

# Module-level storage for last intercepted model and tokenizer
# Used by attention attribution to avoid re-loading
_intercepted_model: Optional[Tuple[Any, Any]] = None


class TransformersInterceptor(BaseInterceptor):
    """Patches HuggingFace transformers to record LLM requests and responses."""

    def __init__(self) -> None:
        self._installed = False
        self._original_pipeline_call: Callable | None = None
        self._original_generate: Callable | None = None

    def install(self) -> None:
        """Install transformers patches."""
        if self._installed:
            return

        try:
            import transformers
        except ImportError:
            return

        # Patch Pipeline.__call__
        try:
            self._original_pipeline_call = transformers.Pipeline.__call__
            transformers.Pipeline.__call__ = self._wrap_pipeline_call(
                self._original_pipeline_call
            )
        except (AttributeError, TypeError):
            pass

        # Patch GenerationMixin.generate
        try:
            from transformers.generation.utils import GenerationMixin

            self._original_generate = GenerationMixin.generate
            GenerationMixin.generate = self._wrap_generate(self._original_generate)
        except (ImportError, AttributeError, TypeError):
            pass

        self._installed = True

    def uninstall(self) -> None:
        """Restore original transformers functions."""
        if not self._installed:
            return

        try:
            import transformers
        except ImportError:
            return

        # Restore Pipeline.__call__
        if self._original_pipeline_call:
            try:
                transformers.Pipeline.__call__ = self._original_pipeline_call
            except (AttributeError, TypeError):
                pass

        # Restore GenerationMixin.generate
        if self._original_generate:
            try:
                from transformers.generation.utils import GenerationMixin

                GenerationMixin.generate = self._original_generate
            except (ImportError, AttributeError, TypeError):
                pass

        self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed

    def _wrap_pipeline_call(self, original: Callable) -> Callable:
        """Wrap Pipeline.__call__ method."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract input text
            input_text = ""
            if args:
                input_text = str(args[0]) if args[0] else ""
            elif "inputs" in kwargs:
                input_text = (
                    str(kwargs["inputs"])
                    if kwargs["inputs"]
                    else ""
                )

            # Extract model name
            model_name = ""
            if hasattr(self_, "model") and hasattr(self_.model, "config"):
                model_name = (
                    self_.model.config.name_or_path
                    or self_.model.config.model_type
                    or ""
                )

            # Store model per-session (not global) to avoid cross-session bleed
            # in concurrent servers where multiple threads use different models.
            if hasattr(self_, "model") and hasattr(self_, "tokenizer"):
                try:
                    session = bus.get_or_create_session()
                    session.hf_model = self_.model
                    session.hf_tokenizer = self_.tokenizer
                except Exception:
                    pass

            # Create request event
            request_event = LLMRequestEvent(
                provider="transformers",
                model=model_name,
                messages=[{"role": "user", "content": input_text}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_length", 1024),
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
                response = original(self_, *args, **kwargs)
            except Exception:
                raise

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""

            # Extract output from response
            if isinstance(response, list) and len(response) > 0:
                item = response[0]
                if isinstance(item, dict):
                    output_text = item.get("generated_text", "")
                elif isinstance(item, str):
                    output_text = item
            elif isinstance(response, str):
                output_text = response

            response_event = LLMResponseEvent(
                request_id=request_event.id,
                output_text=output_text,
                latency_ms=latency_ms,
                prompt_tokens=0,  # Not easily available
                completion_tokens=0,  # Not easily available
                cost_usd=0.0,
            )

            bus.record_llm_response(response_event)

            return response

        return wrapper

    def _wrap_generate(self, original: Callable) -> Callable:
        """Wrap GenerationMixin.generate method."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Extract model name
            model_name = ""
            if hasattr(self_, "config"):
                model_name = (
                    self_.config.name_or_path
                    or self_.config.model_type
                    or ""
                )

            # Store model per-session (not global)
            if hasattr(self_, "tokenizer"):
                try:
                    session = bus.get_or_create_session()
                    session.hf_model = self_
                    session.hf_tokenizer = self_.tokenizer
                except Exception:
                    pass

            # Create request event
            request_event = LLMRequestEvent(
                provider="transformers",
                model=model_name,
                messages=[],  # generate is lower-level, no messages available
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_length", 1024),
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
                output_ids = original(self_, *args, **kwargs)
            except Exception:
                raise

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            completion_tokens = 0

            # Try to decode output
            try:
                if hasattr(self_, "tokenizer"):
                    tokenizer = self_.tokenizer
                    if output_ids is not None:
                        # output_ids could be tensor or list
                        if hasattr(output_ids, "tolist"):
                            ids = output_ids[0].tolist()
                        elif isinstance(output_ids, list):
                            ids = output_ids[0]
                        else:
                            ids = output_ids.tolist()

                        output_text = tokenizer.decode(
                            ids, skip_special_tokens=True
                        )
                        completion_tokens = len(ids)
            except Exception:
                # If decoding fails, use token count if available
                if output_ids is not None and hasattr(output_ids, "shape"):
                    completion_tokens = output_ids.shape[1]

            response_event = LLMResponseEvent(
                request_id=request_event.id,
                output_text=output_text,
                latency_ms=latency_ms,
                prompt_tokens=0,  # Not easily available
                completion_tokens=completion_tokens,
                cost_usd=0.0,
            )

            bus.record_llm_response(response_event)

            return output_ids

        return wrapper
