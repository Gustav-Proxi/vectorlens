"""Interceptor for ChromaDB vector queries."""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import RetrievedChunk, VectorQueryEvent

_logger = logging.getLogger(__name__)


class ChromaInterceptor(BaseInterceptor):
    """Patches ChromaDB to record vector queries."""

    def __init__(self) -> None:
        self._installed = False
        self._install_lock = threading.Lock()
        self._original_query: Callable | None = None

    def install(self) -> None:
        """Install ChromaDB patches."""
        with self._install_lock:
            if self._installed:
                return

            try:
                import chromadb
                from chromadb.api.models.Collection import Collection
            except ImportError:
                return

            # Patch the query method
            self._original_query = Collection.query
            Collection.query = self._wrap_query(self._original_query)

            self._installed = True

    def uninstall(self) -> None:
        """Restore original ChromaDB functions."""
        with self._install_lock:
            if not self._installed:
                return

            try:
                from chromadb.api.models.Collection import Collection
            except ImportError:
                return

            if self._original_query:
                Collection.query = self._original_query

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
            query_texts = kwargs.get("query_texts", [])
            query_embeddings = kwargs.get("query_embeddings", [])
            n_results = kwargs.get("n_results", 10)

            # Determine query text
            query_text = ""
            if query_texts and len(query_texts) > 0:
                query_text = query_texts[0]
            elif query_embeddings and len(query_embeddings) > 0:
                query_text = f"<embedding:{len(query_embeddings[0])}>"

            # Extract collection name
            collection_name = ""
            if hasattr(self_, "name"):
                collection_name = self_.name
            elif hasattr(self_, "_name"):
                collection_name = self_._name

            # Parse results into RetrievedChunk objects
            chunks: list[RetrievedChunk] = []

            if isinstance(result, dict):
                # ChromaDB returns dict with 'ids', 'documents', 'distances', 'metadatas'
                ids = result.get("ids", [[]])[0] if result.get("ids") else []
                documents = result.get("documents", [[]])[0] if result.get("documents") else []
                distances = result.get("distances", [[]])[0] if result.get("distances") else []
                metadatas = result.get("metadatas", [[]])[0] if result.get("metadatas") else []

                for i, chunk_id in enumerate(ids):
                    text = documents[i] if i < len(documents) else ""
                    # For cosine distance, convert to similarity: 1 - distance, clamped to [0, 1]
                    distance = distances[i] if i < len(distances) else 0.0
                    score = max(0.0, min(1.0, 1.0 - distance))
                    metadata = metadatas[i] if i < len(metadatas) else {}

                    chunks.append(
                        RetrievedChunk(
                            chunk_id=str(chunk_id),
                            text=text,
                            score=score,
                            metadata=metadata,
                        )
                    )

            # Create and record the event
            event = VectorQueryEvent(
                db_type="chroma",
                collection=collection_name,
                query_text=query_text,
                top_k=n_results,
                results=chunks,
            )

            try:
                bus.record_vector_query(event)
            except Exception:
                _logger.debug("VectorLens: failed to record vector query", exc_info=True)

            return result

        return wrapper
