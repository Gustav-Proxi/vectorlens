"""Interceptor for FAISS vector queries."""

from __future__ import annotations

import functools
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import RetrievedChunk, VectorQueryEvent


class FAISSInterceptor(BaseInterceptor):
    """Patches FAISS to record vector queries."""

    def __init__(self) -> None:
        self._installed = False
        self._original_search: Callable | None = None

    def install(self) -> None:
        """Install FAISS patches."""
        if self._installed:
            return

        try:
            import faiss
        except ImportError:
            return

        # Patch the Index.search method
        self._original_search = faiss.Index.search
        faiss.Index.search = self._wrap_search(self._original_search)

        self._installed = True

    def uninstall(self) -> None:
        """Restore original FAISS functions."""
        if not self._installed:
            return

        try:
            import faiss
        except ImportError:
            return

        if self._original_search:
            faiss.Index.search = self._original_search

        self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed

    def _wrap_search(self, original: Callable) -> Callable:
        """Wrap the search method."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Call original function
            distances, indices = original(self_, *args, **kwargs)

            # Extract query information
            query_vector = args[0] if len(args) > 0 else None
            k = args[1] if len(args) > 1 else kwargs.get("k", 10)

            # Query embedding as list[float]
            query_embedding: list[float] = []
            if query_vector is not None:
                try:
                    if hasattr(query_vector, "tolist"):
                        # numpy array
                        query_embedding = query_vector[0].tolist()
                    elif isinstance(query_vector, (list, tuple)):
                        # list or tuple of lists
                        if query_vector and isinstance(query_vector[0], (list, tuple)):
                            query_embedding = list(query_vector[0])
                        else:
                            query_embedding = list(query_vector)
                except (IndexError, AttributeError, TypeError):
                    pass

            # Collection name (FAISS doesn't have names, use object id)
            collection_name = f"faiss_index_{id(self_)}"

            # Parse results into RetrievedChunk objects
            chunks: list[RetrievedChunk] = []

            try:
                if hasattr(indices, "tolist"):
                    # numpy array
                    indices_list = indices[0].tolist() if len(indices.shape) > 1 else indices.tolist()
                    distances_list = distances[0].tolist() if len(distances.shape) > 1 else distances.tolist()
                else:
                    indices_list = indices[0] if len(indices) > 0 else []
                    distances_list = distances[0] if len(distances) > 0 else []

                for idx, dist in zip(indices_list, distances_list):
                    # Skip sentinel values (-1 means no result found in FAISS)
                    if idx == -1:
                        continue

                    # Convert distance to similarity score: 1/(1+distance)
                    score = 1.0 / (1.0 + float(dist))

                    chunks.append(
                        RetrievedChunk(
                            chunk_id=str(int(idx)),
                            text="",
                            score=score,
                            metadata={},
                        )
                    )
            except (IndexError, AttributeError, TypeError, ValueError):
                pass

            # Create and record the event
            event = VectorQueryEvent(
                db_type="faiss",
                collection=collection_name,
                query_text="",
                query_embedding=query_embedding,
                top_k=k,
                results=chunks,
            )

            bus.record_vector_query(event)

            return distances, indices

        return wrapper
