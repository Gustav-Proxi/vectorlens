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
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
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


@pytest.mark.integration
def test_compute_no_attentions_returns_zeros(sample_chunks):
    """When model returns no attentions, all scores stay 0.0."""
    import unittest.mock as mock

    attributor = AttentionAttributor()

    # Mock model that returns outputs without attentions
    mock_model = mock.MagicMock()
    mock_model.device = "cpu"
    mock_outputs = mock.MagicMock()
    mock_outputs.attentions = None
    mock_model.return_value = mock_outputs

    mock_tokenizer = mock.MagicMock()
    import torch
    mock_tokenizer.return_value = {"input_ids": torch.ones(1, 10, dtype=torch.long)}

    chunks = attributor.compute(
        mock_model, mock_tokenizer, "prompt", "output", list(sample_chunks)
    )
    assert all(c.attribution_score == 0.0 for c in chunks)


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
    """Returns [] gracefully when torch is not importable."""
    import builtins
    import importlib
    import unittest.mock as mock

    attributor = AttentionAttributor()

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    chunks = [RetrievedChunk(chunk_id="c1", text="hello", score=0.5)]

    with mock.patch("builtins.__import__", side_effect=mock_import):
        # Need to reload to trigger the import inside compute_per_token
        result = attributor.compute_per_token(
            object(), object(), "prompt", "output", chunks
        )

    # Should return [] without raising
    assert result == []
