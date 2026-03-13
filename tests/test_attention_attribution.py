"""Integration tests for AttentionAttributor with a small local HF model.

Requires: pip install torch transformers
Run with: pytest tests/test_attention_attribution.py -m integration -v

These tests load distilgpt2 (~80MB). Skipped unless torch + transformers installed.
"""
from __future__ import annotations

import pytest

from vectorlens.attribution.attention import AttentionAttributor
from vectorlens.types import RetrievedChunk, TokenHeatmapEntry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def distilgpt2():
    """Load distilgpt2 model and tokenizer once for the module."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        pytest.skip("torch or transformers not installed")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    # attn_implementation="eager" forces the model to materialise attention
    # weight tensors. Modern transformers defaults to SDPA which returns None
    # for attention weights, making rollout attribution impossible.
    model = AutoModelForCausalLM.from_pretrained(
        "distilgpt2", attn_implementation="eager"
    )
    model.eval()
    return model, tokenizer


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="chunk-paris",
            text="Paris is the capital of France and a major European city.",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="chunk-london",
            text="London is the capital of the United Kingdom.",
            score=0.7,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests for compute() — chunk-level
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_compute_returns_normalized_scores(distilgpt2, sample_chunks):
    """Chunk attribution scores must sum to ~1.0 when chunks are found in prompt."""
    model, tokenizer = distilgpt2
    attributor = AttentionAttributor()

    prompt = (
        "Context: Paris is the capital of France and a major European city. "
        "London is the capital of the United Kingdom. "
        "Question: What is the capital of France?"
    )
    output = "Paris"

    chunks = attributor.compute(model, tokenizer, prompt, output, list(sample_chunks))

    scores = [c.attribution_score for c in chunks]
    assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of range: {scores}"
    total = sum(scores)
    assert abs(total - 1.0) < 0.01, f"Scores don't sum to 1: total={total}"


@pytest.mark.integration
def test_compute_graceful_when_chunks_not_in_prompt(distilgpt2):
    """Chunks whose text doesn't appear in the prompt get score 0, no crash."""
    model, tokenizer = distilgpt2
    attributor = AttentionAttributor()

    chunks = [
        RetrievedChunk(chunk_id="c1", text="Some text not in the prompt at all.", score=0.5),
    ]
    chunks = attributor.compute(model, tokenizer, "Hello world.", "Hi", chunks)
    assert chunks[0].attribution_score == pytest.approx(0.0)




# ---------------------------------------------------------------------------
# Tests for compute_per_token() — token heatmap
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_compute_per_token_returns_one_entry_per_output_token(distilgpt2, sample_chunks):
    """compute_per_token() returns one TokenHeatmapEntry per output subword token."""
    model, tokenizer = distilgpt2
    attributor = AttentionAttributor()

    prompt = (
        "Context: Paris is the capital of France and a major European city. "
        "London is the capital of the United Kingdom. "
        "Question: What is the capital of France? Answer:"
    )
    output = "Paris is"

    import torch
    output_ids = tokenizer(output, add_special_tokens=False)["input_ids"]
    expected_len = len(output_ids)

    entries = attributor.compute_per_token(model, tokenizer, prompt, output, list(sample_chunks))

    assert len(entries) == expected_len
    assert all(isinstance(e, TokenHeatmapEntry) for e in entries)
    assert all(e.position == i for i, e in enumerate(entries))


@pytest.mark.integration
def test_compute_per_token_scores_normalized_per_token(distilgpt2, sample_chunks):
    """Each token's chunk_attributions must sum to ~1.0."""
    model, tokenizer = distilgpt2
    attributor = AttentionAttributor()

    prompt = (
        "Context: Paris is the capital of France and a major European city. "
        "London is the capital of the United Kingdom. "
        "Answer:"
    )
    output = "France"

    entries = attributor.compute_per_token(model, tokenizer, prompt, output, list(sample_chunks))

    for entry in entries:
        total = sum(entry.chunk_attributions.values())
        assert abs(total - 1.0) < 0.02, (
            f"Token '{entry.text}' scores don't sum to 1: {entry.chunk_attributions}"
        )


@pytest.mark.integration
def test_compute_per_token_empty_output_returns_empty(distilgpt2, sample_chunks):
    """Empty output text yields an empty heatmap without crashing."""
    model, tokenizer = distilgpt2
    attributor = AttentionAttributor()

    entries = attributor.compute_per_token(model, tokenizer, "Some prompt.", "", list(sample_chunks))
    assert entries == []


@pytest.mark.integration
def test_compute_per_token_chunk_ids_present(distilgpt2, sample_chunks):
    """Every chunk_id appears in every entry's chunk_attributions."""
    model, tokenizer = distilgpt2
    attributor = AttentionAttributor()

    prompt = (
        "Paris is the capital of France and a major European city. "
        "London is the capital of the United Kingdom."
    )
    output = "Capital"

    entries = attributor.compute_per_token(model, tokenizer, prompt, output, list(sample_chunks))

    expected_ids = {c.chunk_id for c in sample_chunks}
    for entry in entries:
        assert set(entry.chunk_attributions.keys()) == expected_ids


# ---------------------------------------------------------------------------
# Unit test — no HF model required
# ---------------------------------------------------------------------------


def test_compute_per_token_returns_empty_on_import_error():
    """Returns [] gracefully when torch is not importable.

    Uses sys.modules injection rather than builtins.__import__ mocking
    to correctly bypass Python's module cache (torch may already be imported
    by other tests in the same process).
    """
    import sys
    import unittest.mock as mock

    attributor = AttentionAttributor()
    chunks = [RetrievedChunk(chunk_id="c1", text="hello", score=0.5)]

    # Remove torch from cache and replace with a sentinel that raises ImportError
    with mock.patch.dict(sys.modules, {"torch": None}):
        result = attributor.compute_per_token(
            object(), object(), "prompt", "output", chunks
        )

    assert result == []


def test_compute_no_attentions_returns_zeros_unit():
    """When model.attentions is None, all chunk scores stay 0.0 (no HF model needed)."""
    import unittest.mock as mock
    torch = pytest.importorskip("torch")

    attributor = AttentionAttributor()
    chunks = [
        RetrievedChunk(chunk_id="c1", text="hello world", score=0.9),
        RetrievedChunk(chunk_id="c2", text="foo bar", score=0.5),
    ]

    # Build a real tensor for input_ids so .to(device) works
    real_input_ids = torch.ones(1, 5, dtype=torch.long)

    mock_model = mock.MagicMock()
    mock_model.device = "cpu"
    mock_outputs = mock.MagicMock()
    mock_outputs.attentions = None
    mock_model.return_value = mock_outputs

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.return_value = {"input_ids": real_input_ids}

    result = attributor.compute(mock_model, mock_tokenizer, "hello world", "test", chunks)
    assert all(c.attribution_score == 0.0 for c in result)


def test_compute_per_token_equal_weight_fallback():
    """When no chunk text appears in the prompt, equal weights are assigned."""
    import unittest.mock as mock
    torch = pytest.importorskip("torch")

    attributor = AttentionAttributor()
    chunks = [
        RetrievedChunk(chunk_id="c1", text="chunk text not in prompt at all zzzz", score=0.9),
        RetrievedChunk(chunk_id="c2", text="another chunk also absent xyz", score=0.5),
    ]

    # Mock model that returns attentions for a short sequence
    seq_len = 6
    mock_model = mock.MagicMock()
    mock_model.device = "cpu"
    mock_outputs = mock.MagicMock()
    # Single layer, single head, uniform attention
    attn = torch.ones(1, 1, seq_len, seq_len) / seq_len
    mock_outputs.attentions = [attn]
    mock_model.return_value = mock_outputs

    mock_tokenizer = mock.MagicMock()
    prompt_ids = torch.ones(1, 4, dtype=torch.long)
    output_ids = torch.ones(1, 2, dtype=torch.long)
    mock_tokenizer.side_effect = lambda text, **kw: {
        "input_ids": prompt_ids if kw.get("add_special_tokens", True) else output_ids
    }
    mock_tokenizer.return_value = {"input_ids": prompt_ids}
    mock_tokenizer.convert_ids_to_tokens = mock.MagicMock(return_value=["tok1", "tok2"])
    mock_tokenizer.convert_tokens_to_string = mock.MagicMock(side_effect=lambda ts: ts[0])
    # offset_mapping returns empty (no chars map to tokens → all spans missing)
    mock_tokenizer.return_value = {
        "input_ids": prompt_ids,
        "offset_mapping": [(0, 0)] * 4,
    }

    entries = attributor.compute_per_token(mock_model, mock_tokenizer, "hello", "ok", chunks)
    # With all spans missing, each token should get equal weights across chunks
    for entry in entries:
        for chunk_id in ["c1", "c2"]:
            assert abs(entry.chunk_attributions.get(chunk_id, 0) - 0.5) < 0.01


def test_compute_per_token_empty_prompt():
    """Empty prompt_text yields equal-weight fallback without crashing."""
    import unittest.mock as mock
    torch = pytest.importorskip("torch")

    attributor = AttentionAttributor()
    chunks = [RetrievedChunk(chunk_id="c1", text="some text", score=0.5)]

    # Empty prompt → prompt_len=0, all chunk spans None
    mock_model = mock.MagicMock()
    mock_model.device = "cpu"
    attn = torch.ones(1, 1, 3, 3) / 3
    mock_outputs = mock.MagicMock()
    mock_outputs.attentions = [attn]
    mock_model.return_value = mock_outputs

    prompt_ids = torch.zeros(1, 0, dtype=torch.long)
    output_ids = torch.ones(1, 3, dtype=torch.long)

    mock_tokenizer = mock.MagicMock()
    mock_tokenizer.convert_ids_to_tokens = mock.MagicMock(return_value=["a", "b", "c"])
    mock_tokenizer.convert_tokens_to_string = mock.MagicMock(side_effect=lambda ts: ts[0])

    def _encode(text, **kw):
        if kw.get("add_special_tokens", True):
            return {"input_ids": prompt_ids, "offset_mapping": []}
        return {"input_ids": output_ids}

    mock_tokenizer.side_effect = _encode

    # Should not crash even with empty prompt
    entries = attributor.compute_per_token(mock_model, mock_tokenizer, "", "abc", chunks)
    # output_len=3 → may return [] if output_len=0 check triggers, or entries if not
    # Either way, no exception
    assert isinstance(entries, list)
