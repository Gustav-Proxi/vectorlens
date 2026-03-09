"""Native pgvector interceptor — patches SQLAlchemy execute() to capture vector similarity queries."""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import RetrievedChunk, VectorQueryEvent

_logger = logging.getLogger(__name__)

# pgvector operators (cosine distance, L2 distance, inner product)
_PGVECTOR_OPS = ("<=>", "<->", "<#>")


def _is_vector_query(sql_text: str) -> bool:
    """Check if SQL contains pgvector operators."""
    if not sql_text:
        return False
    return any(op in sql_text for op in _PGVECTOR_OPS)


def _get_sql_string(statement: Any) -> str:
    """Extract SQL string from various SQLAlchemy statement types."""
    if isinstance(statement, str):
        return statement
    # TextClause (from text(...))
    if hasattr(statement, "text"):
        return str(statement.text)
    # Compiled query objects
    try:
        return str(statement)
    except Exception:
        return ""


def _build_event_from_rows(
    rows: list[Any], sql_str: str, latency_ms: float
) -> VectorQueryEvent | None:
    """Extract VectorQueryEvent from query results.

    Tries common column name patterns for ID, text, score, and metadata.
    """
    if not rows:
        return None

    chunks: list[RetrievedChunk] = []
    for row in rows:
        # Convert row to dict — support both RowMapping and namedtuple
        row_dict: dict[str, Any] = {}
        if hasattr(row, "_asdict"):
            # namedtuple
            row_dict = row._asdict()
        elif hasattr(row, "keys"):
            # sqlalchemy.engine.Row (has keys() method)
            try:
                row_dict = dict(row)
            except (TypeError, ValueError):
                # Fallback: manual key extraction
                try:
                    row_dict = {key: row[key] for key in row.keys()}
                except Exception:
                    row_dict = {}
        else:
            # Try generic dict conversion
            try:
                row_dict = dict(row)
            except (TypeError, ValueError):
                row_dict = {}

        # Extract chunk_id (try multiple column names)
        chunk_id = str(
            row_dict.get("id")
            or row_dict.get("chunk_id")
            or row_dict.get("_id")
            or ""
        )

        # Extract text (try multiple column names)
        text = str(
            row_dict.get("text")
            or row_dict.get("content")
            or row_dict.get("page_content")
            or row_dict.get("document")
            or ""
        )

        # Extract score/similarity (try multiple column names)
        score_raw = (
            row_dict.get("score")
            or row_dict.get("similarity")
            or row_dict.get("distance")
            or row_dict.get("_score")
            or 0.0
        )
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # All other columns as metadata (exclude known fields and embeddings)
        metadata = {
            k: str(v)[:200]
            for k, v in row_dict.items()
            if k
            not in (
                "id",
                "chunk_id",
                "_id",
                "text",
                "content",
                "page_content",
                "document",
                "score",
                "similarity",
                "distance",
                "_score",
                "embedding",
            )
        }

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id or str(len(chunks)),
                text=text,
                score=score,
                metadata=metadata,
            )
        )

    return VectorQueryEvent(
        db_type="pgvector",
        collection="postgres",
        query_text="",  # query text not easily extractable from raw SQL
        top_k=len(chunks),
        results=chunks,
    )


class _BufferedResult:
    """Wraps a SQLAlchemy result with pre-fetched rows.

    Allows iteration and common access patterns after the result cursor
    has been consumed.
    """

    def __init__(self, rows: list[Any], original_result: Any) -> None:
        self._rows = rows
        self._original = original_result
        self._index = 0

    def fetchall(self) -> list[Any]:
        """Return all buffered rows."""
        return self._rows

    def fetchone(self) -> Any | None:
        """Return next row or None."""
        if self._index < len(self._rows):
            row = self._rows[self._index]
            self._index += 1
            return row
        return None

    def __iter__(self) -> Any:
        """Iterate over buffered rows."""
        return iter(self._rows)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to original result."""
        return getattr(self._original, name)

    def mappings(self) -> Any:
        """Return a mapping result for dict-like access."""

        class _MappingResult:
            def __init__(self, rows: list[Any]) -> None:
                self._rows = rows

            def fetchall(self) -> list[dict[str, Any]]:
                result = []
                for r in self._rows:
                    if r is None:
                        continue
                    if hasattr(r, "_asdict"):
                        result.append(r._asdict())
                    else:
                        try:
                            result.append(dict(r))
                        except (TypeError, ValueError):
                            pass
                return result

            def __iter__(self) -> Any:
                return iter(self.fetchall())

        return _MappingResult(self._rows)

    @property
    def rowcount(self) -> int:
        """Return number of buffered rows."""
        return len(self._rows)


def _make_async_wrapper(original_execute: Callable) -> Callable:
    """Create async wrapper for AsyncSession.execute."""

    @functools.wraps(original_execute)
    async def patched_async_execute(
        self: Any, statement: Any, parameters: Any = None, **kwargs: Any
    ) -> Any:
        start = time.time()
        result = await original_execute(self, statement, parameters, **kwargs)

        try:
            sql_str = _get_sql_string(statement)
            if not _is_vector_query(sql_str):
                return result

            # Buffer all rows
            rows = result.fetchall()
            latency_ms = (time.time() - start) * 1000

            # Build and record event
            event = _build_event_from_rows(rows, sql_str, latency_ms)
            if event:
                try:
                    bus.record_vector_query(event)
                except Exception:
                    _logger.debug("pgvector: failed to record", exc_info=True)

            # Return wrapped result
            return _BufferedResult(rows, result)
        except Exception:
            # Always return result, even on error
            return result

    return patched_async_execute


def _make_sync_wrapper(original_execute: Callable) -> Callable:
    """Create sync wrapper for Session.execute."""

    @functools.wraps(original_execute)
    def patched_sync_execute(
        self: Any, statement: Any, parameters: Any = None, **kwargs: Any
    ) -> Any:
        start = time.time()
        result = original_execute(self, statement, parameters, **kwargs)

        try:
            sql_str = _get_sql_string(statement)
            if not _is_vector_query(sql_str):
                return result

            # Buffer all rows
            rows = result.fetchall()
            latency_ms = (time.time() - start) * 1000

            # Build and record event
            event = _build_event_from_rows(rows, sql_str, latency_ms)
            if event:
                try:
                    bus.record_vector_query(event)
                except Exception:
                    _logger.debug("pgvector: failed to record", exc_info=True)

            # Return wrapped result
            return _BufferedResult(rows, result)
        except Exception:
            # Always return result, even on error
            return result

    return patched_sync_execute


class PGVectorInterceptor(BaseInterceptor):
    """Patches SQLAlchemy AsyncSession.execute + Session.execute to capture pgvector queries.

    Detects pgvector operators (<=> , <->, <#>) in raw SQL and records vector query events.
    Handles both sync and async SQLAlchemy sessions.
    """

    def __init__(self) -> None:
        self._installed = False
        self._install_lock = threading.Lock()
        self._original_async_execute: Callable | None = None
        self._original_sync_execute: Callable | None = None

    def install(self) -> None:
        """Install pgvector interceptor by patching SQLAlchemy Session.execute methods."""
        with self._install_lock:
            if self._installed:
                return

            try:
                from sqlalchemy.ext.asyncio import AsyncSession

                self._original_async_execute = AsyncSession.execute
                AsyncSession.execute = _make_async_wrapper(self._original_async_execute)
            except ImportError:
                pass

            try:
                from sqlalchemy.orm import Session

                self._original_sync_execute = Session.execute
                Session.execute = _make_sync_wrapper(self._original_sync_execute)
            except ImportError:
                pass

            if self._original_async_execute or self._original_sync_execute:
                self._installed = True

    def uninstall(self) -> None:
        """Restore original SQLAlchemy execute methods."""
        with self._install_lock:
            if not self._installed:
                return

            try:
                from sqlalchemy.ext.asyncio import AsyncSession

                if self._original_async_execute:
                    AsyncSession.execute = self._original_async_execute
            except ImportError:
                pass

            try:
                from sqlalchemy.orm import Session

                if self._original_sync_execute:
                    Session.execute = self._original_sync_execute
            except ImportError:
                pass

            self._installed = False

    def is_installed(self) -> bool:
        """Return True if interceptor is currently installed."""
        return self._installed
