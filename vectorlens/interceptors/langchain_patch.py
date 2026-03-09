"""LangChain interceptor — captures LCEL chain calls and retriever results."""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import LLMRequestEvent, LLMResponseEvent, VectorQueryEvent, RetrievedChunk

_logger = logging.getLogger(__name__)


def _convert_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert LangChain BaseMessage objects to dict format."""
    result = []
    for msg in messages:
        role = getattr(msg, "type", "user")
        content = str(getattr(msg, "content", ""))
        result.append({"role": role, "content": content})
    return result


class LangChainInterceptor(BaseInterceptor):
    """Patches LangChain LCEL pipelines to record chain calls and retriever results."""

    def __init__(self) -> None:
        self._installed = False
        self._install_lock = threading.Lock()
        self._original_generate: Callable | None = None
        self._original_agenerate: Callable | None = None
        self._original_invoke: Callable | None = None
        self._original_ainvoke: Callable | None = None

    def install(self) -> None:
        """Install LangChain patches."""
        with self._install_lock:
            if self._installed:
                return

            try:
                from langchain.chat_models.base import BaseChatModel
                from langchain.schema import BaseRetriever
            except ImportError:
                return

            # Patch BaseChatModel._generate
            self._original_generate = BaseChatModel._generate
            BaseChatModel._generate = self._wrap_generate(self._original_generate)

            # Patch BaseChatModel._agenerate
            self._original_agenerate = BaseChatModel._agenerate
            BaseChatModel._agenerate = self._wrap_agenerate(self._original_agenerate)

            # Patch BaseRetriever.invoke
            self._original_invoke = BaseRetriever.invoke
            BaseRetriever.invoke = self._wrap_invoke(self._original_invoke)

            # Patch BaseRetriever.ainvoke
            self._original_ainvoke = BaseRetriever.ainvoke
            BaseRetriever.ainvoke = self._wrap_ainvoke(self._original_ainvoke)

            self._installed = True

    def uninstall(self) -> None:
        """Restore original LangChain functions."""
        with self._install_lock:
            if not self._installed:
                return

            try:
                from langchain.chat_models.base import BaseChatModel
                from langchain.schema import BaseRetriever
            except ImportError:
                return

            if self._original_generate:
                BaseChatModel._generate = self._original_generate
            if self._original_agenerate:
                BaseChatModel._agenerate = self._original_agenerate
            if self._original_invoke:
                BaseRetriever.invoke = self._original_invoke
            if self._original_ainvoke:
                BaseRetriever.ainvoke = self._original_ainvoke

            self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed

    def _wrap_generate(self, original: Callable) -> Callable:
        """Wrap sync _generate method."""

        @functools.wraps(original)
        def wrapper(self_: Any, messages: list[Any], **kwargs: Any) -> Any:
            # Extract model name
            model = getattr(self_, "model_name", None) or getattr(self_, "model", "langchain")

            # Convert messages
            converted_messages = _convert_messages(messages)

            # Extract parameters
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            # Create request event
            request_event = LLMRequestEvent(
                provider="langchain",
                model=str(model),
                messages=converted_messages,
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
                # Link to parent request if within 30 seconds
                if session.llm_responses:
                    last_response = session.llm_responses[-1]
                    elapsed = time.time() - last_response.timestamp
                    if elapsed < 30.0:
                        request_event.parent_request_id = last_response.request_id
                        request_event.chain_step = "follow_up"
            except Exception:
                _logger.debug("VectorLens: failed to link queries/responses", exc_info=True)

            try:
                bus.record_llm_request(request_event)
            except Exception:
                _logger.debug("VectorLens: failed to record request", exc_info=True)

            # Call original function
            start_time = time.time()
            response = original(self_, messages, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Extract output from LangChain AIMessage
            if hasattr(response, "generations") and response.generations:
                gen = response.generations[0]
                if hasattr(gen, "text"):
                    output_text = gen.text or ""

            if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                llm_out = response.llm_output
                if "token_usage" in llm_out:
                    usage = llm_out["token_usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

            response_event = LLMResponseEvent(
                request_id=request_event.id,
                output_text=output_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            try:
                bus.record_llm_response(response_event)
            except Exception:
                _logger.debug("VectorLens: failed to record response", exc_info=True)

            return response

        return wrapper

    def _wrap_agenerate(self, original: Callable) -> Callable:
        """Wrap async _agenerate method."""

        @functools.wraps(original)
        async def wrapper(self_: Any, messages: list[Any], **kwargs: Any) -> Any:
            # Extract model name
            model = getattr(self_, "model_name", None) or getattr(self_, "model", "langchain")

            # Convert messages
            converted_messages = _convert_messages(messages)

            # Extract parameters
            temperature = kwargs.get("temperature", 0.7)
            max_tokens = kwargs.get("max_tokens", 1024)

            # Create request event
            request_event = LLMRequestEvent(
                provider="langchain",
                model=str(model),
                messages=converted_messages,
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
                # Link to parent request if within 30 seconds
                if session.llm_responses:
                    last_response = session.llm_responses[-1]
                    elapsed = time.time() - last_response.timestamp
                    if elapsed < 30.0:
                        request_event.parent_request_id = last_response.request_id
                        request_event.chain_step = "follow_up"
            except Exception:
                _logger.debug("VectorLens: failed to link queries/responses", exc_info=True)

            try:
                bus.record_llm_request(request_event)
            except Exception:
                _logger.debug("VectorLens: failed to record request", exc_info=True)

            # Call original function
            start_time = time.time()
            response = await original(self_, messages, **kwargs)

            # Record response
            latency_ms = (time.time() - start_time) * 1000
            output_text = ""
            prompt_tokens = 0
            completion_tokens = 0

            # Extract output from LangChain AIMessage
            if hasattr(response, "generations") and response.generations:
                gen = response.generations[0]
                if hasattr(gen, "text"):
                    output_text = gen.text or ""

            if hasattr(response, "llm_output") and isinstance(response.llm_output, dict):
                llm_out = response.llm_output
                if "token_usage" in llm_out:
                    usage = llm_out["token_usage"]
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

            response_event = LLMResponseEvent(
                request_id=request_event.id,
                output_text=output_text,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

            try:
                bus.record_llm_response(response_event)
            except Exception:
                _logger.debug("VectorLens: failed to record response", exc_info=True)

            return response

        return wrapper

    def _wrap_invoke(self, original: Callable) -> Callable:
        """Wrap sync invoke method for retrievers."""

        @functools.wraps(original)
        def wrapper(self_: Any, input: str, **kwargs: Any) -> list[Any]:
            # Call original function
            docs = original(self_, input, **kwargs)

            # Extract retriever name and create VectorQueryEvent
            retriever_type = type(self_).__name__
            results = [
                RetrievedChunk(
                    chunk_id=doc.metadata.get("id", str(i)),
                    text=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                    metadata=doc.metadata,
                )
                for i, doc in enumerate(docs)
            ]

            query_event = VectorQueryEvent(
                db_type="langchain-retriever",
                collection=retriever_type,
                query_text=input,
                results=results,
            )

            try:
                bus.record_vector_query(query_event)
            except Exception:
                _logger.debug("VectorLens: failed to record retriever query", exc_info=True)

            return docs

        return wrapper

    def _wrap_ainvoke(self, original: Callable) -> Callable:
        """Wrap async invoke method for retrievers."""

        @functools.wraps(original)
        async def wrapper(self_: Any, input: str, **kwargs: Any) -> list[Any]:
            # Call original function
            docs = await original(self_, input, **kwargs)

            # Extract retriever name and create VectorQueryEvent
            retriever_type = type(self_).__name__
            results = [
                RetrievedChunk(
                    chunk_id=doc.metadata.get("id", str(i)),
                    text=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                    metadata=doc.metadata,
                )
                for i, doc in enumerate(docs)
            ]

            query_event = VectorQueryEvent(
                db_type="langchain-retriever",
                collection=retriever_type,
                query_text=input,
                results=results,
            )

            try:
                bus.record_vector_query(query_event)
            except Exception:
                _logger.debug("VectorLens: failed to record retriever query", exc_info=True)

            return docs

        return wrapper
