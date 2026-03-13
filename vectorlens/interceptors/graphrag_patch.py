"""Interceptor for Microsoft GraphRAG query engines.

Patches LocalSearchMixedContext and GlobalSearchCommunityContext to capture:
- Local search: source text units → RetrievedChunk objects for LIME attribution
- Global search: community report texts → GraphRAGCommunityUnit objects for
  semantic similarity attribution (solves the "no discrete chunk" problem)

Gracefully skips if graphrag is not installed.
"""

from __future__ import annotations

import functools
import logging
import threading
import uuid
from typing import Any, Callable

from vectorlens.interceptors.base import BaseInterceptor
from vectorlens.session_bus import bus
from vectorlens.types import GraphRAGCommunityUnit, GraphRAGContextEvent, RetrievedChunk

_logger = logging.getLogger(__name__)


class GraphRAGInterceptor(BaseInterceptor):
    """Patches GraphRAG context builders to capture retrieval units."""

    def __init__(self) -> None:
        self._installed = False
        self._lock = threading.Lock()
        self._original_local_build: Callable | None = None
        self._original_global_build: Callable | None = None

    def install(self) -> None:
        with self._lock:
            if self._installed:
                return
            try:
                self._patch_local_search()
                self._patch_global_search()
                self._installed = True
            except ImportError:
                pass  # graphrag not installed — skip silently
            except Exception:
                _logger.debug("GraphRAG interceptor install failed", exc_info=True)

    def uninstall(self) -> None:
        with self._lock:
            if not self._installed:
                return
            try:
                from graphrag.query.structured_search.local_search.mixed_context import (
                    LocalSearchMixedContext,
                )
                from graphrag.query.structured_search.global_search.community_context import (
                    GlobalSearchCommunityContext,
                )

                if self._original_local_build is not None:
                    LocalSearchMixedContext.build_context = self._original_local_build
                if self._original_global_build is not None:
                    GlobalSearchCommunityContext.build_context = self._original_global_build
            except ImportError:
                pass
            self._installed = False

    def is_installed(self) -> bool:
        return self._installed

    # ------------------------------------------------------------------
    # Patching
    # ------------------------------------------------------------------

    def _patch_local_search(self) -> None:
        from graphrag.query.structured_search.local_search.mixed_context import (
            LocalSearchMixedContext,
        )

        self._original_local_build = LocalSearchMixedContext.build_context
        original = self._original_local_build

        @functools.wraps(original)
        async def patched_local(self_: Any, *args: Any, **kwargs: Any) -> Any:
            result = await original(self_, *args, **kwargs)
            try:
                _record_local_context(result, args, kwargs)
            except Exception:
                _logger.debug("Failed to record local GraphRAG context", exc_info=True)
            return result

        LocalSearchMixedContext.build_context = patched_local  # type: ignore[method-assign]

    def _patch_global_search(self) -> None:
        from graphrag.query.structured_search.global_search.community_context import (
            GlobalSearchCommunityContext,
        )

        self._original_global_build = GlobalSearchCommunityContext.build_context
        original = self._original_global_build

        @functools.wraps(original)
        async def patched_global(self_: Any, *args: Any, **kwargs: Any) -> Any:
            result = await original(self_, *args, **kwargs)
            try:
                _record_global_context(result, args, kwargs)
            except Exception:
                _logger.debug("Failed to record global GraphRAG context", exc_info=True)
            return result

        GlobalSearchCommunityContext.build_context = patched_global  # type: ignore[method-assign]


# ------------------------------------------------------------------
# Context extraction helpers (module-level to avoid closure capture)
# ------------------------------------------------------------------

def _record_local_context(result: Any, args: tuple, kwargs: dict) -> None:
    """Extract text units from LocalSearch build_context result and record."""
    context_str, context_data = result if isinstance(result, tuple) else (result, {})

    # Extract query — first positional arg after self is selected_entities,
    # second is query. Or from kwargs.
    query = kwargs.get("query", "")
    if not query and len(args) >= 2:
        query = str(args[1]) if not isinstance(args[0], str) else str(args[0])

    # Extract source text units from context_data
    chunks: list[RetrievedChunk] = []
    if isinstance(context_data, dict):
        sources = context_data.get("sources", [])
        if not sources:
            # Some versions use "text_units"
            sources = context_data.get("text_units", [])

        for i, src in enumerate(sources):
            text = _extract_text(src)
            if text:
                chunks.append(RetrievedChunk(
                    chunk_id=f"graphrag-local-{i}",
                    text=text,
                    score=1.0,  # GraphRAG doesn't expose per-chunk similarity scores
                    metadata={"source": "graphrag_local", "index": i},
                ))

    if not chunks and isinstance(context_str, str) and context_str:
        # Fallback: treat the full context string as one unit (better than nothing)
        chunks.append(RetrievedChunk(
            chunk_id="graphrag-local-full-context",
            text=context_str[:4000],
            score=1.0,
            metadata={"source": "graphrag_local_fallback"},
        ))

    event = GraphRAGContextEvent(
        search_type="local",
        query=query,
        context_text=context_str if isinstance(context_str, str) else "",
        text_chunks=chunks,
    )
    bus.record_graphrag_context(event)
    _logger.debug(
        f"GraphRAG local context captured: {len(chunks)} text units, "
        f"query={query[:50]!r}"
    )


def _record_global_context(result: Any, args: tuple, kwargs: dict) -> None:
    """Extract community reports from GlobalSearch build_context result and record."""
    context_str, context_data = result if isinstance(result, tuple) else (result, {})

    query = kwargs.get("query", "")
    if not query and len(args) >= 1:
        query = str(args[0])

    community_units: list[GraphRAGCommunityUnit] = []

    if isinstance(context_data, dict):
        # GlobalSearchCommunityContext puts reports under various keys
        reports = (
            context_data.get("reports")
            or context_data.get("community_reports")
            or context_data.get("communities")
            or []
        )

        for i, report in enumerate(reports):
            unit = _community_unit_from_report(report, i)
            if unit:
                community_units.append(unit)

    if not community_units and isinstance(context_str, str) and context_str:
        # Fallback: parse community sections from the formatted context string.
        # GraphRAG formats global context as sections separated by "----" or
        # labeled with "## Community X" headers.
        community_units = _parse_community_sections(context_str)

    if not community_units and isinstance(context_str, str) and context_str:
        # Last resort: single unit with full context
        community_units.append(GraphRAGCommunityUnit(
            community_id="0",
            title="Full Context",
            text=context_str[:8000],
            rank=1.0,
        ))

    event = GraphRAGContextEvent(
        search_type="global",
        query=query,
        context_text=context_str if isinstance(context_str, str) else "",
        community_units=community_units,
    )
    bus.record_graphrag_context(event)
    _logger.debug(
        f"GraphRAG global context captured: {len(community_units)} community units, "
        f"query={query[:50]!r}"
    )


def _extract_text(src: Any) -> str:
    """Extract text string from a GraphRAG source object (str, dict, or dataclass)."""
    if isinstance(src, str):
        return src.strip()
    if isinstance(src, dict):
        return (src.get("text") or src.get("content") or src.get("body") or "").strip()
    # dataclass / object with .text attribute
    text = getattr(src, "text", None) or getattr(src, "content", None)
    return str(text).strip() if text else ""


def _community_unit_from_report(report: Any, index: int) -> GraphRAGCommunityUnit | None:
    """Convert a community report object (dict or dataclass) to GraphRAGCommunityUnit."""
    if isinstance(report, str):
        return GraphRAGCommunityUnit(
            community_id=str(index),
            title=f"Community {index}",
            text=report.strip(),
            rank=float(index),  # rank by position (earlier = more relevant)
        )

    if isinstance(report, dict):
        text = (
            report.get("summary")
            or report.get("full_content")
            or report.get("content")
            or report.get("text")
            or ""
        )
        return GraphRAGCommunityUnit(
            community_id=str(report.get("id", index)),
            title=str(report.get("title", f"Community {index}")),
            text=text.strip(),
            rank=float(report.get("rank", index)),
        )

    # Dataclass / object
    text = (
        getattr(report, "summary", None)
        or getattr(report, "full_content", None)
        or getattr(report, "content", None)
        or getattr(report, "text", None)
        or ""
    )
    if not text:
        return None
    return GraphRAGCommunityUnit(
        community_id=str(getattr(report, "id", index)),
        title=str(getattr(report, "title", f"Community {index}")),
        text=str(text).strip(),
        rank=float(getattr(report, "rank", index)),
    )


def _parse_community_sections(context_str: str) -> list[GraphRAGCommunityUnit]:
    """Fallback: parse community sections from formatted GraphRAG context string.

    Handles two common formats:
    1. Sections separated by "----" or "---"
    2. Headers like "## Community X" or "# Community X"
    """
    import re

    units: list[GraphRAGCommunityUnit] = []

    # Try header-based split first (## Community N or # Section N).
    # Use re.split with a pattern that also matches headers at start-of-string.
    sections = re.split(r"(?:^|\n)#{1,3}\s+", context_str)
    if len(sections) > 1:
        for i, section in enumerate(sections[1:], start=1):  # skip any leading preamble
            lines = section.strip().splitlines()
            title = lines[0].strip() if lines else f"Community {i}"
            body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section.strip()
            if body:
                units.append(GraphRAGCommunityUnit(
                    community_id=str(i),
                    title=title,
                    text=body,
                    rank=float(i),
                ))
        return units

    # Try separator-based split (--- or more dashes, possibly surrounded by blank lines)
    sections = re.split(r"\n\s*-{3,}\s*\n", context_str)
    if len(sections) > 1:
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) > 20:
                lines = section.splitlines()
                title = lines[0].strip() if lines else f"Section {i}"
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else section
                units.append(GraphRAGCommunityUnit(
                    community_id=str(i),
                    title=title,
                    text=body or section,
                    rank=float(i),
                ))

    return units
