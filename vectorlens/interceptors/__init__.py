"""Monkey-patch interceptors for LLM providers and vector databases.

This module provides a registry of interceptors that can be installed/uninstalled
to record LLM requests/responses and vector database queries (ChromaDB, Pinecone,
FAISS, Weaviate, pgvector, etc.).
"""

from __future__ import annotations

import threading
from typing import Any

from vectorlens.interceptors.anthropic_patch import AnthropicInterceptor
from vectorlens.interceptors.chroma_patch import ChromaInterceptor
from vectorlens.interceptors.faiss_patch import FAISSInterceptor
from vectorlens.interceptors.gemini_patch import GeminiInterceptor
from vectorlens.interceptors.httpx_transport import HttpxTransportInterceptor
from vectorlens.interceptors.langchain_patch import LangChainInterceptor
from vectorlens.interceptors.openai_patch import OpenAIInterceptor
from vectorlens.interceptors.pgvector_patch import PGVectorInterceptor
from vectorlens.interceptors.pinecone_patch import PineconeInterceptor
from vectorlens.interceptors.transformers_patch import TransformersInterceptor
from vectorlens.interceptors.weaviate_patch import WeaviateInterceptor

# Global registry of all available interceptors
_INTERCEPTORS: dict[str, Any] = {
    "httpx": HttpxTransportInterceptor(),
    "openai": OpenAIInterceptor(),
    "anthropic": AnthropicInterceptor(),
    "gemini": GeminiInterceptor(),
    "langchain": LangChainInterceptor(),
    "chroma": ChromaInterceptor(),
    "pinecone": PineconeInterceptor(),
    "faiss": FAISSInterceptor(),
    "transformers": TransformersInterceptor(),
    "weaviate": WeaviateInterceptor(),
    "pgvector": PGVectorInterceptor(),
}

_lock = threading.Lock()


def install_all() -> list[str]:
    """Install all available interceptors.

    Silently skips interceptors whose libraries are not installed.

    Returns:
        List of names of successfully installed interceptors.
    """
    with _lock:
        installed = []
        for name, interceptor in _INTERCEPTORS.items():
            try:
                interceptor.install()
                if interceptor.is_installed():
                    installed.append(name)
            except Exception:
                # Silently skip if installation fails (e.g., library not installed)
                pass
        return installed


def uninstall_all() -> None:
    """Uninstall all interceptors and restore original functions."""
    with _lock:
        for interceptor in _INTERCEPTORS.values():
            try:
                interceptor.uninstall()
            except Exception:
                # Silently ignore errors during uninstall
                pass


def get_installed() -> list[str]:
    """Return list of currently installed interceptors."""
    with _lock:
        return [name for name, interceptor in _INTERCEPTORS.items() if interceptor.is_installed()]


__all__ = [
    "install_all",
    "uninstall_all",
    "get_installed",
]
