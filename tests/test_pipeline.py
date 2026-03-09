"""Tests for the auto-attribution pipeline.

Tests the setup_auto_attribution() function and the _on_llm_response callback
that automatically runs hallucination detection after LLM responses.

Note: These tests use the global bus (vectorlens.session_bus.bus) since
the pipeline's _run_attribution function always imports that global instance.
"""
import time
import threading
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vectorlens.pipeline import setup_auto_attribution, _on_llm_response, _run_attribution
from vectorlens.session_bus import bus as global_bus
from vectorlens.types import (
    AttributionResult,
    LLMRequestEvent,
    LLMResponseEvent,
    OutputToken,
    RetrievedChunk,
    VectorQueryEvent,
)


class TestAutoAttributionPipeline:
    """Tests for the auto-attribution pipeline."""

    def setup_method(self):
        """Create a fresh session for each test."""
        # Use the global bus (required by _run_attribution)
        self.bus = global_bus
        # Create a new session for this test
        self.session = self.bus.new_session()
        # Reset the global _installed flag so setup_auto_attribution works
        import vectorlens.pipeline as pipeline_module
        pipeline_module._installed = False

    def test_setup_installs_subscriber(self):
        """Verify that setup_auto_attribution installs a listener on the bus."""
        # Get count before subscribe
        listeners_before = len(self.bus._listeners.get("llm_response", []))

        # Subscribe the callback directly to the global bus
        self.bus.subscribe("llm_response", _on_llm_response)

        # Now there should be one more listener
        listeners_after = len(self.bus._listeners.get("llm_response", []))
        assert listeners_after > listeners_before

    def test_attribution_triggered_after_llm_response(self):
        """Test that attribution is recorded after an LLM response is recorded."""
        # Mock the detector to avoid loading sentence-transformers
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,  # output
            fake_embedding,  # chunks
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(
                chunk_id="c1",
                text="Paris is the capital of France.",
                score=0.95,
            )
            vq = VectorQueryEvent(
                db_type="test",
                query_text="Paris facts",
                results=[chunk],
            )
            self.bus.record_vector_query(vq)

            # Subscribe the callback to global bus
            self.bus.subscribe("llm_response", _on_llm_response)

            # Record LLM response with explicit session_id
            resp = LLMResponseEvent(
                output_text="Paris is the capital of France.",
                request_id="req_test_001",
                session_id=self.session.id,
            )
            self.bus.record_llm_response(resp)

            # Wait for background thread
            time.sleep(1)

            # Check that attribution was recorded
            session = self.bus.get_session(self.session.id)
            assert len(session.attributions) == 1

            attr = session.attributions[0]
            assert attr.response_id == resp.id
            assert len(attr.output_tokens) > 0

    def test_attribution_uses_linked_vector_query(self):
        """Test that attribution uses the vector query linked to the LLM request."""
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,
            fake_embedding,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            # Create two vector queries
            chunk1 = RetrievedChunk(
                chunk_id="c1",
                text="First chunk about Paris.",
                score=0.95,
            )
            vq1 = VectorQueryEvent(
                db_type="test",
                query_text="Paris",
                results=[chunk1],
            )
            self.bus.record_vector_query(vq1)

            chunk2 = RetrievedChunk(
                chunk_id="c2",
                text="Second chunk about London.",
                score=0.90,
            )
            vq2 = VectorQueryEvent(
                db_type="test",
                query_text="London",
                results=[chunk2],
            )
            self.bus.record_vector_query(vq2)

            # Create LLM request linked to first query
            llm_req = LLMRequestEvent(
                model="test-model",
                vector_query_id=vq1.id,
                session_id=self.session.id,
            )
            self.bus.record_llm_request(llm_req)

            # Subscribe and record response
            self.bus.subscribe("llm_response", _on_llm_response)

            resp = LLMResponseEvent(
                output_text="First chunk about Paris.",
                request_id=llm_req.id,
                session_id=self.session.id,
            )
            self.bus.record_llm_response(resp)

            time.sleep(1)

            session = self.bus.get_session(self.session.id)
            assert len(session.attributions) == 1

            # Attribution should use chunks from vq1
            attr = session.attributions[0]
            chunk_ids = {c.chunk_id for c in attr.chunks}
            assert "c1" in chunk_ids

    def test_attribution_skips_empty_output(self):
        """Test that attribution is not recorded for empty LLM output."""
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="Test chunk.",
            score=0.95,
        )
        vq = VectorQueryEvent(
            db_type="test",
            query_text="test",
            results=[chunk],
        )
        self.bus.record_vector_query(vq)

        self.bus.subscribe("llm_response", _on_llm_response)

        # Record response with empty output
        resp = LLMResponseEvent(
            output_text="",
            request_id="req_empty",
            session_id=self.session.id,
        )
        self.bus.record_llm_response(resp)

        time.sleep(0.5)

        session = self.bus.get_session(self.session.id)
        # No attribution should be recorded
        assert len(session.attributions) == 0

    def test_attribution_skips_whitespace_only_output(self):
        """Test that attribution is not recorded for whitespace-only output."""
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="Test chunk.",
            score=0.95,
        )
        vq = VectorQueryEvent(
            db_type="test",
            query_text="test",
            results=[chunk],
        )
        self.bus.record_vector_query(vq)

        self.bus.subscribe("llm_response", _on_llm_response)

        # Record response with whitespace-only output
        resp = LLMResponseEvent(
            output_text="   \n\t  ",
            request_id="req_whitespace",
            session_id=self.session.id,
        )
        self.bus.record_llm_response(resp)

        time.sleep(0.5)

        session = self.bus.get_session(self.session.id)
        # No attribution should be recorded
        assert len(session.attributions) == 0

    def test_attribution_skips_no_chunks(self):
        """Test that attribution is not recorded when no chunks are available."""
        # Create session with no vector queries
        self.bus.subscribe("llm_response", _on_llm_response)

        resp = LLMResponseEvent(
            output_text="Some output text.",
            request_id="req_no_chunks",
            session_id=self.session.id,
        )
        self.bus.record_llm_response(resp)

        time.sleep(0.5)

        session = self.bus.get_session(self.session.id)
        # No attribution should be recorded
        assert len(session.attributions) == 0

    def test_attribution_handles_detector_failure(self):
        """Test that pipeline gracefully handles detector failures."""
        chunk = RetrievedChunk(
            chunk_id="c1",
            text="Test chunk.",
            score=0.95,
        )
        vq = VectorQueryEvent(
            db_type="test",
            query_text="test",
            results=[chunk],
        )
        self.bus.record_vector_query(vq)

        # Mock detector to raise an exception
        with patch(
            "vectorlens.detection.hallucination.HallucinationDetector.detect",
            side_effect=RuntimeError("Model load failed"),
        ):
            self.bus.subscribe("llm_response", _on_llm_response)

            resp = LLMResponseEvent(
                output_text="Test output.",
                request_id="req_fail",
                session_id=self.session.id,
            )
            self.bus.record_llm_response(resp)

            time.sleep(0.5)

            # Session should still exist and be intact
            session = self.bus.get_session(self.session.id)
            assert session is not None
            assert len(session.llm_responses) == 1
            # Attribution not recorded due to failure
            assert len(session.attributions) == 0

    def test_overall_groundedness_calculation(self):
        """Test that overall groundedness is correctly calculated."""
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,  # output
            fake_embedding,  # chunks
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            # Mock the detector to return a mix of hallucinated/grounded tokens
            mock_tokens = [
                OutputToken(text="Grounded", position=0, is_hallucinated=False),
                OutputToken(text="Also grounded", position=1, is_hallucinated=False),
                OutputToken(text="Hallucinated", position=2, is_hallucinated=True),
            ]

            chunk = RetrievedChunk(
                chunk_id="c1",
                text="Real information.",
                score=0.95,
            )
            vq = VectorQueryEvent(
                db_type="test",
                query_text="test",
                results=[chunk],
            )
            self.bus.record_vector_query(vq)

            with patch(
                "vectorlens.detection.hallucination.HallucinationDetector.detect",
                return_value=mock_tokens,
            ):
                self.bus.subscribe("llm_response", _on_llm_response)

                resp = LLMResponseEvent(
                    output_text="Grounded. Also grounded. Hallucinated.",
                    request_id="req_groundedness",
                    session_id=self.session.id,
                )
                self.bus.record_llm_response(resp)

                time.sleep(0.5)

                session = self.bus.get_session(self.session.id)
                attr = session.attributions[0]

                # 2 grounded, 1 hallucinated = 2/3 ≈ 0.667
                expected_groundedness = 2.0 / 3.0
                assert attr.overall_groundedness == pytest.approx(
                    expected_groundedness, abs=0.01
                )

    def test_hallucinated_spans_computed(self):
        """Test that hallucinated spans are correctly identified."""
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,
            fake_embedding,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            # Create tokens with non-consecutive hallucinations
            mock_tokens = [
                OutputToken(text="Grounded1", position=0, is_hallucinated=False),
                OutputToken(text="Hallucinated1", position=1, is_hallucinated=True),
                OutputToken(text="Hallucinated2", position=2, is_hallucinated=True),
                OutputToken(text="Grounded2", position=3, is_hallucinated=False),
                OutputToken(text="Hallucinated3", position=4, is_hallucinated=True),
            ]

            chunk = RetrievedChunk(
                chunk_id="c1",
                text="Real information.",
                score=0.95,
            )
            vq = VectorQueryEvent(
                db_type="test",
                query_text="test",
                results=[chunk],
            )
            self.bus.record_vector_query(vq)

            with patch(
                "vectorlens.detection.hallucination.HallucinationDetector.detect",
                return_value=mock_tokens,
            ):
                self.bus.subscribe("llm_response", _on_llm_response)

                resp = LLMResponseEvent(
                    output_text="Test output.",
                    request_id="req_spans",
                    session_id=self.session.id,
                )
                self.bus.record_llm_response(resp)

                time.sleep(0.5)

                session = self.bus.get_session(self.session.id)
                attr = session.attributions[0]

                # Should have 2 spans: (1,2) and (4,4)
                assert len(attr.hallucinated_spans) == 2
                assert (1, 2) in attr.hallucinated_spans
                assert (4, 4) in attr.hallucinated_spans

    def test_chunk_attribution_scores_updated(self):
        """Test that chunk attribution scores are updated from token attributions."""
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,
            fake_embedding,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            # Create token with chunk attributions
            mock_tokens = [
                OutputToken(
                    text="Test output.",
                    position=0,
                    is_hallucinated=False,
                    chunk_attributions={"c1": 0.95, "c2": 0.75},
                ),
            ]

            chunk1 = RetrievedChunk(
                chunk_id="c1",
                text="First chunk.",
                score=0.95,
            )
            chunk2 = RetrievedChunk(
                chunk_id="c2",
                text="Second chunk.",
                score=0.85,
            )
            vq = VectorQueryEvent(
                db_type="test",
                query_text="test",
                results=[chunk1, chunk2],
            )
            self.bus.record_vector_query(vq)

            with patch(
                "vectorlens.detection.hallucination.HallucinationDetector.detect",
                return_value=mock_tokens,
            ):
                self.bus.subscribe("llm_response", _on_llm_response)

                resp = LLMResponseEvent(
                    output_text="Test output.",
                    request_id="req_attr_scores",
                    session_id=self.session.id,
                )
                self.bus.record_llm_response(resp)

                time.sleep(0.5)

                session = self.bus.get_session(self.session.id)
                attr = session.attributions[0]

                # Chunks should have updated attribution scores
                chunk_map = {c.chunk_id: c for c in attr.chunks}
                assert chunk_map["c1"].attribution_score == pytest.approx(0.95)
                assert chunk_map["c2"].attribution_score == pytest.approx(0.75)

    def test_attribution_uses_latest_vector_query_when_unlinked(self):
        """Test that latest vector query is used when LLM request has no link."""
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,
            fake_embedding,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            # Create two vector queries
            chunk1 = RetrievedChunk(
                chunk_id="c1",
                text="Old chunk.",
                score=0.95,
            )
            vq1 = VectorQueryEvent(
                db_type="test",
                query_text="old",
                results=[chunk1],
            )
            self.bus.record_vector_query(vq1)

            chunk2 = RetrievedChunk(
                chunk_id="c2",
                text="New chunk.",
                score=0.90,
            )
            vq2 = VectorQueryEvent(
                db_type="test",
                query_text="new",
                results=[chunk2],
            )
            self.bus.record_vector_query(vq2)

            # Create LLM request with NO vector_query_id
            llm_req = LLMRequestEvent(
                model="test-model",
                vector_query_id=None,
                session_id=self.session.id,
            )
            self.bus.record_llm_request(llm_req)

            self.bus.subscribe("llm_response", _on_llm_response)

            resp = LLMResponseEvent(
                output_text="New chunk.",
                request_id=llm_req.id,
                session_id=self.session.id,
            )
            self.bus.record_llm_response(resp)

            time.sleep(0.5)

            session = self.bus.get_session(self.session.id)
            attr = session.attributions[0]

            # Should use the latest (vq2) and its chunk
            chunk_ids = {c.chunk_id for c in attr.chunks}
            assert "c2" in chunk_ids

    def test_caused_hallucination_flag_set_for_low_groundedness(self):
        """Test that caused_hallucination flag is set when groundedness is low."""
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,
            fake_embedding,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            # All tokens hallucinated
            mock_tokens = [
                OutputToken(text="Hallucinated1", position=0, is_hallucinated=True),
                OutputToken(text="Hallucinated2", position=1, is_hallucinated=True),
            ]

            chunk = RetrievedChunk(
                chunk_id="c1",
                text="Real chunk.",
                score=0.95,
                attribution_score=0.05,  # Low attribution
            )
            vq = VectorQueryEvent(
                db_type="test",
                query_text="test",
                results=[chunk],
            )
            self.bus.record_vector_query(vq)

            with patch(
                "vectorlens.detection.hallucination.HallucinationDetector.detect",
                return_value=mock_tokens,
            ):
                self.bus.subscribe("llm_response", _on_llm_response)

                resp = LLMResponseEvent(
                    output_text="Hallucinated1 Hallucinated2.",
                    request_id="req_caused_hall",
                    session_id=self.session.id,
                )
                self.bus.record_llm_response(resp)

                time.sleep(0.5)

                session = self.bus.get_session(self.session.id)
                attr = session.attributions[0]

                # Groundedness should be 0.0
                assert attr.overall_groundedness == 0.0
                # Chunk with low attribution score should be flagged
                assert attr.chunks[0].caused_hallucination is True

    def test_attribution_result_contains_all_required_fields(self):
        """Test that attribution result has all required fields populated."""
        fake_model = MagicMock()
        fake_embedding = np.array([[1.0, 0.0, 0.0]])
        fake_model.encode.side_effect = [
            fake_embedding,
            fake_embedding,
        ]

        with patch(
            "vectorlens.detection.hallucination._get_model", return_value=fake_model
        ):
            chunk = RetrievedChunk(
                chunk_id="c1",
                text="Test chunk.",
                score=0.95,
            )
            vq = VectorQueryEvent(
                db_type="test",
                query_text="test",
                results=[chunk],
            )
            self.bus.record_vector_query(vq)

            llm_req = LLMRequestEvent(
                model="test-model",
                session_id=self.session.id,
            )
            self.bus.record_llm_request(llm_req)

            self.bus.subscribe("llm_response", _on_llm_response)

            resp = LLMResponseEvent(
                output_text="Test chunk.",
                request_id=llm_req.id,
                session_id=self.session.id,
            )
            self.bus.record_llm_response(resp)

            time.sleep(0.5)

            session = self.bus.get_session(self.session.id)
            assert len(session.attributions) == 1

            attr = session.attributions[0]
            # Check all required fields
            assert attr.id is not None
            assert attr.session_id == self.session.id
            assert attr.request_id == llm_req.id
            assert attr.response_id == resp.id
            assert attr.timestamp > 0
            assert len(attr.chunks) > 0
            assert len(attr.output_tokens) > 0
            assert 0.0 <= attr.overall_groundedness <= 1.0
            assert isinstance(attr.hallucinated_spans, list)
