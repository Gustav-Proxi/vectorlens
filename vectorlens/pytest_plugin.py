"""VectorLens pytest plugin.

Provides the `vectorlens_session` fixture and `@pytest.mark.min_groundedness`
marker for asserting RAG groundedness in tests without running a server.

Usage:
    # conftest.py
    pytest_plugins = ['vectorlens.pytest_plugin']

    # test_rag.py
    def test_answer_is_grounded(vectorlens_session):
        response = my_rag(query="What is attention?")
        assert vectorlens_session.groundedness >= 0.8

    @pytest.mark.min_groundedness(0.8)
    def test_answer_grounded_via_marker(vectorlens_session):
        my_rag(query="What is attention?")
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import pytest

from vectorlens.interceptors import install_all, uninstall_all
from vectorlens.session_bus import SessionBus
from vectorlens.pipeline import setup_auto_attribution, _run_attribution


_GROUNDEDNESS_TIMEOUT = 10.0  # seconds
_POLL_INTERVAL = 0.1


class VectorLensSession:
    """Test session object with blocking .groundedness property."""

    def __init__(self, session_id: str, bus: SessionBus) -> None:
        self._session_id = session_id
        self._bus = bus

    @property
    def id(self) -> str:
        return self._session_id

    @property
    def groundedness(self) -> float:
        """Block until attribution completes (max 10s), then return score."""
        deadline = time.monotonic() + _GROUNDEDNESS_TIMEOUT
        while time.monotonic() < deadline:
            session = self._bus.get_session(self._session_id)
            if session and session.attributions:
                return session.attributions[-1].overall_groundedness
            time.sleep(_POLL_INTERVAL)
        raise TimeoutError(
            f"VectorLens attribution did not complete within {_GROUNDEDNESS_TIMEOUT}s. "
            "Ensure your RAG code makes an LLM call with retrieved chunks."
        )

    @property
    def attributions(self):
        session = self._bus.get_session(self._session_id)
        return session.attributions if session else []

    @property
    def hallucinated_count(self) -> int:
        session = self._bus.get_session(self._session_id)
        if not session or not session.attributions:
            return 0
        tokens = session.attributions[-1].output_tokens
        return sum(1 for t in tokens if t.is_hallucinated)


def pytest_configure(config: pytest.Config) -> None:
    """Register the min_groundedness marker."""
    config.addinivalue_line(
        "markers",
        "min_groundedness(threshold): fail test if RAG groundedness < threshold",
    )


@pytest.fixture
def vectorlens_session():
    """Fixture that installs VectorLens interceptors and yields a session.

    Interceptors are installed before the test and removed after.
    No server is started — attribution runs in-process.
    """
    bus = SessionBus()
    session = bus.new_session()
    vl_session = VectorLensSession(session.id, bus)

    # Wire attribution pipeline to the local bus
    def _on_response(event):
        _run_attribution(event, _bus=bus)

    bus.subscribe("llm_response", _on_response)
    install_all()

    yield vl_session

    uninstall_all()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """Apply @pytest.mark.min_groundedness threshold check after test runs."""
    outcome = yield

    marker = item.get_closest_marker("min_groundedness")
    if marker is None:
        return

    threshold = marker.args[0] if marker.args else 0.8
    session_fixture = item.funcargs.get("vectorlens_session")
    if session_fixture is None:
        raise pytest.UsageError(
            "@pytest.mark.min_groundedness requires the `vectorlens_session` fixture"
        )

    try:
        score = session_fixture.groundedness
    except TimeoutError as e:
        pytest.fail(str(e))
        return

    if score < threshold:
        pytest.fail(
            f"RAG groundedness {score:.2f} is below required threshold {threshold:.2f}. "
            f"Hallucinated sentences: {session_fixture.hallucinated_count}"
        )
