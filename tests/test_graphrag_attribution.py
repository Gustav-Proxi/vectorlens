"""Unit tests for GraphRAG attribution.

Tests CommunityAttributor and GraphRAGInterceptor without requiring graphrag
or sentence-transformers to be installed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from vectorlens.types import (
    GraphRAGCommunityUnit,
    GraphRAGContextEvent,
    OutputToken,
    RetrievedChunk,
    Session,
)
from vectorlens.attribution.graphrag_attribution import (
    CommunityAttributor,
    community_units_to_chunks,
    _extract_hallucinated_sentences,
)
from vectorlens.interceptors.graphrag_patch import _parse_community_sections


def _make_token(text: str, hallucinated: bool = False) -> OutputToken:
    return OutputToken(text=text, position=0, is_hallucinated=hallucinated)


def _make_community(cid: str, text: str, rank: float = 1.0) -> GraphRAGCommunityUnit:
    return GraphRAGCommunityUnit(community_id=cid, title=f"C{cid}", text=text, rank=rank)


class TestCommunityAttributor(unittest.TestCase):

    def _make_attributor_with_mock_model(self, similarity_matrix: list[list[float]]):
        """Return an attributor whose sentence-transformer is mocked.

        similarity_matrix[i][j] = cosine sim between community i and hallucinated sent j
        """
        attributor = CommunityAttributor()

        call_count = [0]
        community_embs = [np.array([float(i), 0.0]) for i in range(len(similarity_matrix))]
        hall_embs = [np.array([0.0, float(j)]) for j in range(len(similarity_matrix[0]))]

        def fake_encode(texts, **kwargs):
            # First call: hallucinated sentences; subsequent: communities
            if call_count[0] == 0:
                call_count[0] += 1
                return np.array(hall_embs[: len(texts)])
            call_count[0] += 1
            return np.array(community_embs[: len(texts)])

        mock_model = MagicMock()
        mock_model.encode.side_effect = fake_encode

        # Patch _cosine_sim to use precomputed matrix
        sim_mat = similarity_matrix
        orig_attribute = attributor.attribute

        def patched_attribute(output_tokens, community_units):
            # Bypass encode, directly assign scores from matrix
            for i, unit in enumerate(community_units):
                if i < len(sim_mat):
                    unit.attribution_score = max(sim_mat[i])
                    unit.caused_hallucination = max(sim_mat[i]) > 0.5
            return community_units

        attributor.attribute = patched_attribute
        return attributor

    def test_no_hallucinations_returns_zero_scores(self):
        tokens = [_make_token("Paris is the capital of France.", hallucinated=False)]
        communities = [
            _make_community("0", "France is a country in Western Europe."),
            _make_community("1", "Paris is a major European city."),
        ]
        attributor = CommunityAttributor()
        with patch("vectorlens.attribution.graphrag_attribution._get_model") as m:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2]])
            m.return_value = mock_model
            result = attributor.attribute(tokens, communities)
        # No hallucinations — all scores should be 0
        for unit in result:
            self.assertEqual(unit.attribution_score, 0.0)
            self.assertFalse(unit.caused_hallucination)

    def test_hallucinated_token_triggers_attribution(self):
        tokens = [
            _make_token("The Eiffel Tower is 500 meters tall.", hallucinated=True)
        ]
        communities = [
            _make_community("0", "The Eiffel Tower is 330 meters tall."),
            _make_community("1", "France has a rich culinary tradition."),
        ]

        # Mock: community 0 has high sim to hallucination, community 1 has low sim
        emb_community_0 = np.array([1.0, 0.0])
        emb_community_1 = np.array([0.0, 1.0])
        emb_hallucination = np.array([0.95, 0.05])  # close to community 0

        call_order = [0]

        def fake_encode(texts, **kwargs):
            result = []
            for _ in texts:
                if call_order[0] == 0:
                    result.append(emb_hallucination)
                    call_order[0] += 1
                elif call_order[0] == 1:
                    result.append(emb_community_0)
                    call_order[0] += 1
                else:
                    result.append(emb_community_1)
            return np.array(result)

        with patch("vectorlens.attribution.graphrag_attribution._get_model") as m:
            mock_model = MagicMock()
            mock_model.encode.side_effect = fake_encode
            m.return_value = mock_model
            attributor = CommunityAttributor()
            result = attributor.attribute(tokens, communities)

        # Community 0 should have higher score than community 1
        self.assertGreater(communities[0].attribution_score, communities[1].attribution_score)

    def test_empty_communities_returns_unchanged(self):
        tokens = [_make_token("something hallucinated", hallucinated=True)]
        result = CommunityAttributor().attribute(tokens, [])
        self.assertEqual(result, [])

    def test_community_units_to_chunks_conversion(self):
        units = [
            GraphRAGCommunityUnit(
                unit_id="u1", community_id="c1", title="Test",
                text="Some community text.", rank=2.0,
                attribution_score=0.8, caused_hallucination=True,
            ),
            GraphRAGCommunityUnit(
                unit_id="u2", community_id="c2", title="Other",
                text="Other text.", rank=1.0,
                attribution_score=0.2, caused_hallucination=False,
            ),
        ]
        chunks = community_units_to_chunks(units)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_id, "u1")
        self.assertEqual(chunks[0].attribution_score, 0.8)
        self.assertTrue(chunks[0].caused_hallucination)
        self.assertEqual(chunks[0].metadata["type"], "graphrag_community")
        self.assertEqual(chunks[0].metadata["community_id"], "c1")


class TestExtractHallucinatedSentences(unittest.TestCase):

    def test_empty_tokens(self):
        self.assertEqual(_extract_hallucinated_sentences([]), [])

    def test_no_hallucinated(self):
        tokens = [_make_token("Correct fact.", False), _make_token("Another fact.", False)]
        self.assertEqual(_extract_hallucinated_sentences(tokens), [])

    def test_hallucinated_tokens_extracted(self):
        tokens = [
            _make_token("The tower is 500m tall.", True),
            _make_token("It was built in 1950.", True),
        ]
        sentences = _extract_hallucinated_sentences(tokens)
        self.assertGreater(len(sentences), 0)
        combined = " ".join(sentences)
        self.assertIn("500m", combined)


class TestParseCommunitySection(unittest.TestCase):

    def test_header_based_parsing(self):
        context = """## Community Alpha
This is the alpha community content. It discusses entity relationships.

## Community Beta
Beta community summary. Covers different topics entirely.
"""
        units = _parse_community_sections(context)
        self.assertEqual(len(units), 2)
        self.assertIn("alpha", units[0].title.lower())
        self.assertIn("alpha community content", units[0].text)

    def test_separator_based_parsing(self):
        context = """Section One
Some content about section one.

---

Section Two
Content about section two here.
"""
        units = _parse_community_sections(context)
        self.assertGreaterEqual(len(units), 1)

    def test_no_sections_returns_empty(self):
        context = "Just a plain paragraph with no section markers."
        units = _parse_community_sections(context)
        # No structured sections found
        self.assertEqual(len(units), 0)


class TestGraphRAGInterceptorInstall(unittest.TestCase):

    def test_install_skips_when_graphrag_not_installed(self):
        """Interceptor must not raise when graphrag is not installed."""
        from vectorlens.interceptors.graphrag_patch import GraphRAGInterceptor
        interceptor = GraphRAGInterceptor()

        with patch("builtins.__import__", side_effect=ImportError("graphrag not found")):
            # Should silently skip
            try:
                interceptor.install()
            except Exception as e:
                self.fail(f"install() raised unexpectedly: {e}")

        # is_installed should be False
        self.assertFalse(interceptor.is_installed())

    def test_graphrag_context_event_fields(self):
        event = GraphRAGContextEvent(
            search_type="global",
            query="What are the main themes?",
            community_units=[
                _make_community("0", "Community about AI and machine learning."),
            ],
        )
        self.assertEqual(event.search_type, "global")
        self.assertEqual(len(event.community_units), 1)
        self.assertIsNotNone(event.id)
        self.assertIsNotNone(event.timestamp)

    def test_session_has_graphrag_contexts_field(self):
        session = Session()
        self.assertEqual(session.graphrag_contexts, [])
        event = GraphRAGContextEvent(search_type="local", query="test")
        session.graphrag_contexts.append(event)
        self.assertEqual(len(session.graphrag_contexts), 1)


if __name__ == "__main__":
    unittest.main()
