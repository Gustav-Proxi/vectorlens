"""Tests for interceptor modules.

Tests the OpenAI, Anthropic, and ChromaDB interceptors with mocked libraries.
"""
import time
import unittest.mock as mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vectorlens.interceptors import get_installed, install_all, uninstall_all
from vectorlens.interceptors.openai_patch import OpenAIInterceptor
from vectorlens.interceptors.anthropic_patch import AnthropicInterceptor
from vectorlens.interceptors.chroma_patch import ChromaInterceptor
from vectorlens.interceptors.pgvector_patch import PGVectorInterceptor
from vectorlens.session_bus import SessionBus
from vectorlens.types import RetrievedChunk, VectorQueryEvent


# ============================================================================
# OpenAI Interceptor Tests
# ============================================================================


class TestOpenAIInterceptor:
    """Tests for OpenAI API interceptor."""

    def setup_method(self):
        """Fresh session bus and interceptor for each test."""
        self.bus = SessionBus()
        self.interceptor = OpenAIInterceptor()

    def teardown_method(self):
        """Clean up interceptor state."""
        if self.interceptor._installed:
            self.interceptor.uninstall()

    def test_install_uninstall(self):
        """Test that install() patches and uninstall() restores."""
        # Completions is imported lazily inside install(), so patch at source
        mock_completions_cls = MagicMock()
        mock_completions_cls.create = MagicMock()

        mock_module = MagicMock()
        mock_module.Completions = mock_completions_cls

        with patch.dict("sys.modules", {
            "openai": MagicMock(),
            "openai.resources.chat.completions": mock_module,
        }):
            assert not self.interceptor.is_installed()
            self.interceptor.install()
            assert self.interceptor.is_installed()
            self.interceptor.uninstall()
            assert not self.interceptor.is_installed()

    def test_records_llm_request_on_call(self):
        """Test that LLM request is recorded with correct fields."""
        # Create a mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_original_create = MagicMock(return_value=mock_response)

        # Patch the bus to use our test bus
        with patch("vectorlens.interceptors.openai_patch.bus", self.bus):
            # Create wrapper around our mock
            wrapped = self.interceptor._wrap_create(mock_original_create)

            # Call it
            mock_self = MagicMock()
            result = wrapped(
                mock_self,
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0.7,
                max_tokens=1024,
            )

            # Check that request was recorded
            session = self.bus.get_or_create_session()
            assert len(session.llm_requests) == 1
            request = session.llm_requests[0]
            assert request.provider == "openai"
            assert request.model == "gpt-4o"
            assert request.messages == [{"role": "user", "content": "Hello"}]
            assert request.temperature == 0.7
            assert request.max_tokens == 1024

    def test_records_llm_response_with_cost(self):
        """Test that LLM response is recorded with cost calculation."""
        # Create mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Test response"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        mock_original_create = MagicMock(return_value=mock_response)

        with patch("vectorlens.interceptors.openai_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_create(mock_original_create)
            mock_self = MagicMock()

            wrapped(
                mock_self,
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
            )

            session = self.bus.get_or_create_session()
            assert len(session.llm_responses) == 1
            response = session.llm_responses[0]
            assert response.output_text == "Test response"
            assert response.prompt_tokens == 100
            assert response.completion_tokens == 50
            # For gpt-4o: (100 * 5/1M) + (50 * 15/1M)
            expected_cost = (100 * 5 / 1_000_000) + (50 * 15 / 1_000_000)
            assert response.cost_usd == pytest.approx(expected_cost)
            assert response.cost_usd > 0

    def test_skips_gracefully_if_openai_not_installed(self):
        """Test that install() skips gracefully if openai not importable."""
        # Patch import to fail
        with patch("builtins.__import__", side_effect=ImportError("openai not found")):
            # Should not raise
            self.interceptor.install()
            # Should not be installed
            assert not self.interceptor.is_installed()

    def test_links_to_recent_vector_query(self):
        """Test that LLM request is linked to recent vector query."""
        # Record a vector query first
        vector_event = VectorQueryEvent(
            db_type="chroma",
            collection="test",
            query_text="test query",
            top_k=5,
            results=[],
        )

        with patch("vectorlens.interceptors.openai_patch.bus", self.bus):
            self.bus.record_vector_query(vector_event)

            # Now make LLM request shortly after
            mock_response = MagicMock()
            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            mock_response.choices = [mock_choice]
            mock_response.usage.prompt_tokens = 10
            mock_response.usage.completion_tokens = 5

            mock_original_create = MagicMock(return_value=mock_response)
            wrapped = self.interceptor._wrap_create(mock_original_create)
            mock_self = MagicMock()

            # Make request within 5 seconds
            wrapped(mock_self, model="gpt-4o", messages=[])

            session = self.bus.get_or_create_session()
            request = session.llm_requests[0]
            # Should be linked
            assert request.vector_query_id == vector_event.id

    def test_async_request_recording(self):
        """Test that async create also records requests."""
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Async response"
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10

        mock_original_acreate = AsyncMock(return_value=mock_response)

        with patch("vectorlens.interceptors.openai_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_acreate(mock_original_acreate)
            mock_self = MagicMock()

            # Run async — asyncio.run() creates a fresh context, so the
            # wrapper will create its own session via get_or_create_session()
            import asyncio
            asyncio.run(
                wrapped(
                    mock_self,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "Test"}],
                )
            )

            # Find whichever session received the events
            session = next(
                (s for s in self.bus.all_sessions() if s.llm_requests), None
            )
            assert session is not None
            assert len(session.llm_requests) == 1
            assert len(session.llm_responses) == 1
            assert session.llm_responses[0].output_text == "Async response"


# ============================================================================
# Anthropic Interceptor Tests
# ============================================================================


class TestAnthropicInterceptor:
    """Tests for Anthropic API interceptor."""

    def setup_method(self):
        """Fresh session bus and interceptor for each test."""
        self.bus = SessionBus()
        self.interceptor = AnthropicInterceptor()

    def teardown_method(self):
        """Clean up interceptor state."""
        if self.interceptor._installed:
            self.interceptor.uninstall()

    def test_records_anthropic_request(self):
        """Test that Anthropic request is recorded."""
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Anthropic response"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 8

        mock_original_create = MagicMock(return_value=mock_response)

        with patch("vectorlens.interceptors.anthropic_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_create(mock_original_create)
            mock_self = MagicMock()

            wrapped(
                mock_self,
                model="claude-3-5-sonnet",
                messages=[{"role": "user", "content": "Hi"}],
            )

            session = self.bus.get_or_create_session()
            assert len(session.llm_requests) == 1
            assert session.llm_requests[0].provider == "anthropic"
            assert session.llm_requests[0].model == "claude-3-5-sonnet"
            assert session.llm_responses[0].output_text == "Anthropic response"

    def test_anthropic_cost_calculation(self):
        """Test cost calculation for different Anthropic models."""
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Response"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 1000
        mock_response.usage.output_tokens = 500

        mock_original_create = MagicMock(return_value=mock_response)

        with patch("vectorlens.interceptors.anthropic_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_create(mock_original_create)
            mock_self = MagicMock()

            # Test claude-3-5-sonnet
            wrapped(mock_self, model="claude-3-5-sonnet", messages=[])

            session = self.bus.get_or_create_session()
            response = session.llm_responses[0]
            # (1000 * 3/1M) + (500 * 15/1M)
            expected = (1000 * 3 / 1_000_000) + (500 * 15 / 1_000_000)
            assert response.cost_usd == pytest.approx(expected)

    def test_anthropic_async_recording(self):
        """Test async Anthropic recording."""
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "Async anthropic"
        mock_response.content = [mock_content]
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3

        mock_original_acreate = AsyncMock(return_value=mock_response)

        with patch("vectorlens.interceptors.anthropic_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_acreate(mock_original_acreate)
            mock_self = MagicMock()

            import asyncio
            asyncio.run(
                wrapped(
                    mock_self,
                    model="claude-3-haiku",
                    messages=[],
                )
            )

            session = next(
                (s for s in self.bus.all_sessions() if s.llm_responses), None
            )
            assert session is not None
            assert session.llm_responses[0].output_text == "Async anthropic"


# ============================================================================
# ChromaDB Interceptor Tests
# ============================================================================


class TestChromaInterceptor:
    """Tests for ChromaDB vector query interceptor."""

    def setup_method(self):
        """Fresh session bus and interceptor for each test."""
        self.bus = SessionBus()
        self.interceptor = ChromaInterceptor()

    def teardown_method(self):
        """Clean up interceptor state."""
        if self.interceptor._installed:
            self.interceptor.uninstall()

    def test_records_vector_query(self):
        """Test that vector queries are recorded."""
        # Mock ChromaDB result
        mock_result = {
            "ids": [["chunk1", "chunk2"]],
            "documents": [["Document 1", "Document 2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"source": "doc1.txt"}, {"source": "doc2.txt"}]],
        }

        mock_original_query = MagicMock(return_value=mock_result)

        with patch("vectorlens.interceptors.chroma_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_query(mock_original_query)

            # Mock collection object
            mock_collection = MagicMock()
            mock_collection.name = "test_collection"

            result = wrapped(
                mock_collection,
                query_texts=["test query"],
                n_results=10,
            )

            session = self.bus.get_or_create_session()
            assert len(session.vector_queries) == 1

            event = session.vector_queries[0]
            assert event.db_type == "chroma"
            assert event.collection == "test_collection"
            assert event.query_text == "test query"
            assert event.top_k == 10
            assert len(event.results) == 2

    def test_converts_distance_to_score(self):
        """Test that cosine distance is converted to similarity score."""
        # For cosine distance: score = 1 - distance
        mock_result = {
            "ids": [["id1"]],
            "documents": [["text1"]],
            "distances": [[0.2]],  # distance = 0.2
            "metadatas": [[]],
        }

        mock_original_query = MagicMock(return_value=mock_result)

        with patch("vectorlens.interceptors.chroma_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_query(mock_original_query)
            mock_collection = MagicMock()
            mock_collection.name = "coll"

            wrapped(mock_collection, query_texts=["q"])

            session = self.bus.get_or_create_session()
            event = session.vector_queries[0]
            # score should be 1 - 0.2 = 0.8
            assert event.results[0].score == pytest.approx(0.8)

    def test_handles_missing_documents(self):
        """Test handling of missing/None documents in results."""
        # Result with empty documents
        mock_result = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", None]],  # Second is None
            "distances": [[0.1, 0.2]],
            "metadatas": [[{}, {}]],
        }

        mock_original_query = MagicMock(return_value=mock_result)

        with patch("vectorlens.interceptors.chroma_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_query(mock_original_query)
            mock_collection = MagicMock()
            mock_collection.name = "coll"

            # Should not raise
            wrapped(mock_collection, query_texts=["q"])

            session = self.bus.get_or_create_session()
            event = session.vector_queries[0]
            assert len(event.results) == 2
            assert event.results[0].text == "doc1"
            # None should be converted to empty string or preserved
            assert event.results[1].text is None or event.results[1].text == ""

    def test_query_with_embedding(self):
        """Test recording query with embedding vector instead of text."""
        mock_result = {
            "ids": [["id1"]],
            "documents": [["text"]],
            "distances": [[0.1]],
            "metadatas": [[]],
        }

        mock_original_query = MagicMock(return_value=mock_result)

        with patch("vectorlens.interceptors.chroma_patch.bus", self.bus):
            wrapped = self.interceptor._wrap_query(mock_original_query)
            mock_collection = MagicMock()
            mock_collection.name = "coll"

            embedding = [0.1, 0.2, 0.3]
            wrapped(
                mock_collection,
                query_embeddings=[embedding],
                n_results=5,
            )

            session = self.bus.get_or_create_session()
            event = session.vector_queries[0]
            # Should record as embedding marker
            assert "embedding:" in event.query_text


# ============================================================================
# Registry Tests
# ============================================================================


def test_install_all_returns_installed_names():
    """Test that install_all() returns list of successfully installed interceptors."""
    # Mock all libraries as available
    mock_openai = MagicMock()
    mock_anthropic = MagicMock()
    mock_chroma = MagicMock()

    with patch.dict("sys.modules", {
        "openai": mock_openai,
        "openai.resources.chat.completions": MagicMock(),
        "anthropic": mock_anthropic,
        "anthropic.resources.messages": MagicMock(),
        "chromadb": mock_chroma,
        "chromadb.api.models.Collection": MagicMock(),
    }):
        # This will try to import, so we need to mock the classes too
                    installed = install_all()
                    # Should return a list of strings
                    assert isinstance(installed, list)
                    for name in installed:
                        assert isinstance(name, str)


def test_get_installed_reflects_state():
    """Test that get_installed() reflects actual interceptor state."""
    # Before any install
    uninstalled = get_installed()
    assert isinstance(uninstalled, list)

    # After install_all (mocked)
    initial_count = len(get_installed())
    uninstall_all()
    final_count = len(get_installed())
    assert final_count <= initial_count


def test_interceptor_thread_safety():
    """Test that install/uninstall are thread-safe."""
    import threading

    results = []

    def install_and_check():
        try:
            install_all()
            installed = get_installed()
            results.append(("install", installed))
        except Exception as e:
            results.append(("install_error", str(e)))

    def uninstall_and_check():
        try:
            uninstall_all()
            results.append(("uninstall", None))
        except Exception as e:
            results.append(("uninstall_error", str(e)))

    # Run concurrent operations
    threads = [
        threading.Thread(target=install_and_check),
        threading.Thread(target=uninstall_and_check),
        threading.Thread(target=install_and_check),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should complete without errors (even if some operations are no-ops)
    errors = [r for r in results if "error" in r[0]]
    # Thread safety ensures no exceptions are raised
    assert len(errors) == 0


# ============================================================================
# pgvector Interceptor Tests
# ============================================================================


class TestPGVectorInterceptor:
    """Tests for pgvector vector query interceptor."""

    def setup_method(self):
        """Fresh session bus and interceptor for each test."""
        self.bus = SessionBus()
        self.interceptor = PGVectorInterceptor()

    def teardown_method(self):
        """Clean up interceptor state."""
        if self.interceptor._installed:
            self.interceptor.uninstall()

    def test_detects_vector_query_with_cosine_operator(self):
        """Test detection of pgvector cosine distance operator (<=>)."""
        from vectorlens.interceptors.pgvector_patch import _is_vector_query

        sql = "SELECT id, text, 1 - (embedding <=> :vec) AS score FROM chunks LIMIT 10"
        assert _is_vector_query(sql) is True

    def test_detects_vector_query_with_l2_operator(self):
        """Test detection of pgvector L2 distance operator (<->)."""
        from vectorlens.interceptors.pgvector_patch import _is_vector_query

        sql = "SELECT id, (embedding <-> :vec) AS distance FROM chunks LIMIT 5"
        assert _is_vector_query(sql) is True

    def test_detects_vector_query_with_inner_product_operator(self):
        """Test detection of pgvector inner product operator (<#>)."""
        from vectorlens.interceptors.pgvector_patch import _is_vector_query

        sql = "SELECT id, (embedding <#> :vec) FROM chunks"
        assert _is_vector_query(sql) is True

    def test_ignores_non_vector_query(self):
        """Test that non-pgvector queries are ignored."""
        from vectorlens.interceptors.pgvector_patch import _is_vector_query

        sql = "SELECT id, text FROM chunks WHERE id = :id"
        assert _is_vector_query(sql) is False

    def test_extracts_sql_from_text_statement(self):
        """Test extraction of SQL from sqlalchemy text() clause."""
        from vectorlens.interceptors.pgvector_patch import _get_sql_string

        # Test with string
        sql_str = _get_sql_string("SELECT * FROM table")
        assert sql_str == "SELECT * FROM table"

    def test_builds_event_from_rows_with_common_columns(self):
        """Test building VectorQueryEvent from rows with standard column names."""
        from vectorlens.interceptors.pgvector_patch import _build_event_from_rows

        # Create mock rows as namedtuples
        from collections import namedtuple

        Row = namedtuple("Row", ["id", "text", "score"])
        rows = [
            Row(id="chunk1", text="Document 1", score=0.95),
            Row(id="chunk2", text="Document 2", score=0.75),
        ]

        event = _build_event_from_rows(rows, "SELECT ...", 10.0)
        assert event is not None
        assert event.db_type == "pgvector"
        assert event.top_k == 2
        assert len(event.results) == 2
        assert event.results[0].chunk_id == "chunk1"
        assert event.results[0].text == "Document 1"
        assert event.results[0].score == 0.95
        assert event.results[1].chunk_id == "chunk2"

    def test_builds_event_with_alternative_column_names(self):
        """Test column name flexibility (chunk_id, content, similarity, etc.)."""
        from vectorlens.interceptors.pgvector_patch import _build_event_from_rows
        from collections import namedtuple

        Row = namedtuple("Row", ["chunk_id", "content", "similarity"])
        rows = [Row(chunk_id="id1", content="Text 1", similarity=0.85)]

        event = _build_event_from_rows(rows, "SELECT ...", 5.0)
        assert event is not None
        assert event.results[0].chunk_id == "id1"
        assert event.results[0].text == "Text 1"
        assert event.results[0].score == 0.85

    def test_clamps_score_to_range(self):
        """Test that scores outside [0,1] are clamped."""
        from vectorlens.interceptors.pgvector_patch import _build_event_from_rows
        from collections import namedtuple

        Row = namedtuple("Row", ["id", "text", "score"])
        rows = [
            Row(id="id1", text="Text", score=-0.5),  # Below 0
            Row(id="id2", text="Text", score=1.5),   # Above 1
        ]

        event = _build_event_from_rows(rows, "SELECT ...", 5.0)
        assert event.results[0].score == 0.0
        assert event.results[1].score == 1.0

    def test_buffers_result_rows(self):
        """Test that _BufferedResult correctly buffers and provides access to rows."""
        from vectorlens.interceptors.pgvector_patch import _BufferedResult
        from collections import namedtuple

        Row = namedtuple("Row", ["id", "text"])
        rows = [Row(id="1", text="a"), Row(id="2", text="b")]

        mock_original = MagicMock()
        buffered = _BufferedResult(rows, mock_original)

        # fetchall should return all rows
        all_rows = buffered.fetchall()
        assert len(all_rows) == 2
        assert all_rows[0].id == "1"

        # iteration should work
        ids = [row.id for row in buffered]
        assert ids == ["1", "2"]

        # rowcount should be accurate
        assert buffered.rowcount == 2

    def test_buffered_result_fetchone(self):
        """Test fetchone() on _BufferedResult."""
        from vectorlens.interceptors.pgvector_patch import _BufferedResult
        from collections import namedtuple

        Row = namedtuple("Row", ["id"])
        rows = [Row(id="1"), Row(id="2")]

        buffered = _BufferedResult(rows, MagicMock())
        assert buffered.fetchone().id == "1"
        assert buffered.fetchone().id == "2"
        assert buffered.fetchone() is None

    def test_install_and_uninstall(self):
        """Test install/uninstall lifecycle."""
        assert not self.interceptor.is_installed()
        self.interceptor.install()
        # install() succeeds even if sqlalchemy is not available
        # (it catches ImportError silently)
        self.interceptor.uninstall()
        assert not self.interceptor.is_installed()

    def test_handles_missing_sqlalchemy(self):
        """Test graceful handling when sqlalchemy is not installed."""
        # Simulate missing sqlalchemy by patching import
        with patch("builtins.__import__", side_effect=ImportError("sqlalchemy not found")):
            # install() should not raise
            self.interceptor.install()
            # And should not be installed (since sqlalchemy unavailable)
            assert not self.interceptor.is_installed()
