"""Interceptor for Weaviate vector queries."""

from __future__ import annotations

import functools
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import RetrievedChunk, VectorQueryEvent


class WeaviateInterceptor(BaseInterceptor):
    """Patches Weaviate to record vector queries."""

    def __init__(self) -> None:
        self._installed = False
        self._original_near_text: Callable | None = None
        self._original_near_vector: Callable | None = None

    def install(self) -> None:
        """Install Weaviate patches."""
        if self._installed:
            return

        try:
            from weaviate.collections.queries.near_text import NearTextQuery
            from weaviate.collections.queries.near_vector import NearVectorQuery
        except ImportError:
            return

        # Patch NearTextQuery.near_text
        self._original_near_text = NearTextQuery.near_text
        NearTextQuery.near_text = self._wrap_near_text(self._original_near_text)

        # Patch NearVectorQuery.near_vector
        self._original_near_vector = NearVectorQuery.near_vector
        NearVectorQuery.near_vector = self._wrap_near_vector(self._original_near_vector)

        self._installed = True

    def uninstall(self) -> None:
        """Restore original Weaviate functions."""
        if not self._installed:
            return

        try:
            from weaviate.collections.queries.near_text import NearTextQuery
            from weaviate.collections.queries.near_vector import NearVectorQuery
        except ImportError:
            return

        if self._original_near_text:
            NearTextQuery.near_text = self._original_near_text

        if self._original_near_vector:
            NearVectorQuery.near_vector = self._original_near_vector

        self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is installed."""
        return self._installed

    def _wrap_near_text(self, original: Callable) -> Callable:
        """Wrap the near_text method."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Call original function
            result = original(self_, *args, **kwargs)

            # Extract query information
            query_text = kwargs.get("query", "")
            limit = kwargs.get("limit", 10)

            # Get collection name
            collection_name = "unknown"
            try:
                if hasattr(self_, "_name"):
                    collection_name = self_._name
                elif hasattr(self_, "name"):
                    collection_name = self_.name
            except (AttributeError, TypeError):
                pass

            # Parse results into RetrievedChunk objects
            chunks: list[RetrievedChunk] = []

            try:
                if hasattr(result, "objects"):
                    for obj in result.objects:
                        chunk_id = ""
                        if hasattr(obj, "uuid"):
                            chunk_id = str(obj.uuid)

                        # Extract text from properties
                        text = ""
                        properties = {}
                        if hasattr(obj, "properties") and isinstance(obj.properties, dict):
                            properties = obj.properties
                            text = properties.get("text", "")

                        # Extract similarity score from metadata
                        score = 0.0
                        try:
                            if hasattr(obj, "metadata") and hasattr(obj.metadata, "distance"):
                                # Weaviate distance -> similarity: 1 - distance
                                distance = obj.metadata.distance
                                score = 1.0 - float(distance) if distance is not None else 0.0
                        except (AttributeError, TypeError, ValueError):
                            pass

                        chunks.append(
                            RetrievedChunk(
                                chunk_id=chunk_id,
                                text=text,
                                score=score,
                                metadata=properties,
                            )
                        )
            except (AttributeError, TypeError):
                pass

            # Create and record the event
            event = VectorQueryEvent(
                db_type="weaviate",
                collection=collection_name,
                query_text=query_text,
                query_embedding=[],
                top_k=limit,
                results=chunks,
            )

            bus.record_vector_query(event)

            return result

        return wrapper

    def _wrap_near_vector(self, original: Callable) -> Callable:
        """Wrap the near_vector method."""

        @functools.wraps(original)
        def wrapper(self_: Any, *args: Any, **kwargs: Any) -> Any:
            # Call original function
            result = original(self_, *args, **kwargs)

            # Extract query information
            near_vector = kwargs.get("near_vector", None)
            limit = kwargs.get("limit", 10)

            # Query embedding as list[float]
            query_embedding: list[float] = []
            if near_vector is not None:
                try:
                    if isinstance(near_vector, (list, tuple)):
                        query_embedding = list(near_vector)
                    elif hasattr(near_vector, "tolist"):  # numpy array
                        query_embedding = near_vector.tolist()
                except (AttributeError, TypeError):
                    pass

            # Get collection name
            collection_name = "unknown"
            try:
                if hasattr(self_, "_name"):
                    collection_name = self_._name
                elif hasattr(self_, "name"):
                    collection_name = self_.name
            except (AttributeError, TypeError):
                pass

            # Parse results into RetrievedChunk objects
            chunks: list[RetrievedChunk] = []

            try:
                if hasattr(result, "objects"):
                    for obj in result.objects:
                        chunk_id = ""
                        if hasattr(obj, "uuid"):
                            chunk_id = str(obj.uuid)

                        # Extract text from properties
                        text = ""
                        properties = {}
                        if hasattr(obj, "properties") and isinstance(obj.properties, dict):
                            properties = obj.properties
                            text = properties.get("text", "")

                        # Extract similarity score from metadata
                        score = 0.0
                        try:
                            if hasattr(obj, "metadata") and hasattr(obj.metadata, "distance"):
                                # Weaviate distance -> similarity: 1 - distance
                                distance = obj.metadata.distance
                                score = 1.0 - float(distance) if distance is not None else 0.0
                        except (AttributeError, TypeError, ValueError):
                            pass

                        chunks.append(
                            RetrievedChunk(
                                chunk_id=chunk_id,
                                text=text,
                                score=score,
                                metadata=properties,
                            )
                        )
            except (AttributeError, TypeError):
                pass

            # Create and record the event
            event = VectorQueryEvent(
                db_type="weaviate",
                collection=collection_name,
                query_text="",
                query_embedding=query_embedding,
                top_k=limit,
                results=chunks,
            )

            bus.record_vector_query(event)

            return result

        return wrapper
