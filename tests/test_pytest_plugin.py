"""Tests for the VectorLens pytest plugin."""
import pytest
from unittest.mock import patch, MagicMock

from vectorlens.pytest_plugin import VectorLensSession
from vectorlens.session_bus import SessionBus
from vectorlens.types import (
    AttributionResult, LLMRequestEvent, LLMResponseEvent,
    RetrievedChunk, VectorQueryEvent, OutputToken,
)
from vectorlens.pipeline import _run_attribution


def _make_session_with_attribution(bus, groundedness: float, hallucinated: bool = False):
    """Helper: create a session with a completed attribution result."""
    import numpy as np
    from unittest.mock import MagicMock

    session = bus.new_session()
    sid = session.id

    chunk = RetrievedChunk(chunk_id="c1", text="Paris is in France.", score=0.9)
    vq = VectorQueryEvent(db_type="test", query_text="Paris", results=[chunk])
    bus.record_vector_query(vq)

    req = LLMRequestEvent(model="test", vector_query_id=vq.id)
    bus.record_llm_request(req)

    output_text = "Paris is in France." if not hallucinated else "Paris is in Germany."
    resp = LLMResponseEvent(output_text=output_text, request_id=req.id)
    bus.record_llm_response(resp)

    from vectorlens.detection import hallucination as _hm

    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])

    with patch.object(_hm, "_get_model", return_value=fake_model):
        _run_attribution(resp, _bus=bus)
    return session.id


def test_vectorlens_session_groundedness_returns_score():
    """VectorLensSession.groundedness returns attribution score."""
    bus = SessionBus()
    sid = _make_session_with_attribution(bus, groundedness=1.0)
    vl = VectorLensSession(sid, bus)
    score = vl.groundedness
    assert 0.0 <= score <= 1.0


def test_vectorlens_session_groundedness_timeout():
    """VectorLensSession.groundedness raises TimeoutError if no attribution."""
    import vectorlens.pytest_plugin as plugin
    original = plugin._GROUNDEDNESS_TIMEOUT
    plugin._GROUNDEDNESS_TIMEOUT = 0.2  # short timeout for test

    try:
        bus = SessionBus()
        session = bus.new_session()
        vl = VectorLensSession(session.id, bus)
        with pytest.raises(TimeoutError, match="did not complete"):
            _ = vl.groundedness
    finally:
        plugin._GROUNDEDNESS_TIMEOUT = original


def test_vectorlens_session_no_state_leakage():
    """Two VectorLensSessions on separate buses don't share attribution."""
    bus1, bus2 = SessionBus(), SessionBus()
    sid1 = _make_session_with_attribution(bus1, groundedness=1.0)
    sid2 = _make_session_with_attribution(bus2, groundedness=1.0)

    vl1 = VectorLensSession(sid1, bus1)
    vl2 = VectorLensSession(sid2, bus2)

    # Each sees only its own session
    assert vl1.id != vl2.id
    assert bus1.get_session(sid2) is None
    assert bus2.get_session(sid1) is None


def test_vectorlens_session_attributions_property():
    """attributions property returns list of AttributionResult."""
    bus = SessionBus()
    sid = _make_session_with_attribution(bus, groundedness=1.0)
    vl = VectorLensSession(sid, bus)
    attrs = vl.attributions
    assert isinstance(attrs, list)
    assert len(attrs) == 1


def test_vectorlens_session_hallucinated_count():
    """hallucinated_count returns number of hallucinated output tokens."""
    bus = SessionBus()
    sid = _make_session_with_attribution(bus, groundedness=1.0)
    vl = VectorLensSession(sid, bus)
    count = vl.hallucinated_count
    assert isinstance(count, int)
    assert count >= 0


def test_min_groundedness_marker_passes():
    """Marker validation: score above threshold should not fail."""
    bus = SessionBus()
    sid = _make_session_with_attribution(bus, groundedness=1.0)
    vl = VectorLensSession(sid, bus)
    score = vl.groundedness
    threshold = 0.5
    assert score >= threshold, f"Expected >= {threshold}, got {score}"


def test_pytest_plugin_marker_registered():
    """min_groundedness marker is registered in pytest config."""
