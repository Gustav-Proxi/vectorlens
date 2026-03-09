"""Interceptor for Pinecone vector queries."""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import RetrievedChunk, VectorQueryEvent

_logger = logging.getLogger(__name__)


class PineconeInterceptor(BaseInterceptor):
    """Patches Pinecone to record vector queries."""

    def __init__(self) -> None:
        self._installed = False
        self._install_lock = threading.Lock()
        self._original_query: Callable | None = None

    def install(self) -> None:
        """Install Pinecone patches."""
        with self._install_lock:
            if self._installed:
                return

            try:
                import pinecone
            except ImportError:
                return

            # Patch the Index.query method
            self._original_query = pinecone.Index.query
            pinecone.Index.query = self._wrap_query(self._original_query)

            self._installed = True

    def uninstall(self) -> None:
        """Restore original Pinecone functions."""
        with self._install_lock:
            if not self._installed:
                return

            try:
                import pinecone
            except ImportError:
                return

            if self._original_query:
                pinecone.Index.query = self._original_query

            self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed

    def _wrap_query(self, original: Callable) -> Callable:
        """Wrap the query method."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Call original function
            result = original(self_, *args, **kwargs)

            # Extract query information
            vector = kwargs.get("vector", None)
            if vector is None and len(args) > 0:
                vector = args[0]

            top_k = kwargs.get("top_k", 10)
            if top_k is None and len(args) > 1:
                top_k = args[1]

            # Get collection name (index name)
            collection_name = "unknown"
            try:
                # Try _config.index_name first
                if hasattr(self_, "_config") and hasattr(self_._config, "index_name"):
                    collection_name = self_._config.index_name
                # Fall back to index_name attribute
                elif hasattr(self_, "index_name"):
                    collection_name = self_.index_name
            except (AttributeError, TypeError):
                pass

            # Query embedding as list[float]
            query_embedding: list[float] = []
            if vector is not None:
                if isinstance(vector, (list, tuple)):
                    query_embedding = list(vector)
                elif hasattr(vector, "tolist"):  # numpy array
                    query_embedding = vector.tolist()

            # Parse results into RetrievedChunk objects
            chunks: list[RetrievedChunk] = []

            if hasattr(result, "matches"):
                for match in result.matches:
                    chunk_id = getattr(match, "id", "")
                    # Pinecone scores are cosine similarity [0,1], clamp defensively
                    score = max(0.0, min(1.0, float(getattr(match, "score", 0.0) or 0.0)))

                    # Extract text from metadata
                    metadata = getattr(match, "metadata", None) or {}
                    text = metadata.get("text", "") if isinstance(metadata, dict) else ""

                    chunks.append(
                        RetrievedChunk(
                            chunk_id=str(chunk_id),
                            text=text,
                            score=score,
                            metadata=metadata if isinstance(metadata, dict) else {},
                        )
                    )

            # Create and record the event
            event = VectorQueryEvent(
                db_type="pinecone",
                collection=collection_name,
                query_text="",
                query_embedding=query_embedding,
                top_k=top_k,
                results=chunks,
            )

            try:
                bus.record_vector_query(event)
            except Exception:
                _logger.debug("VectorLens: failed to record vector query", exc_info=True)

            return result

        return wrapper
