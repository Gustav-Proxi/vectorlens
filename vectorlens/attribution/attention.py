"""Token-level attribution via attention rollout for HuggingFace models.

Attention rollout: multiply attention matrices across layers to get
effective attention from each output token to each input token.
This tells us: "which input tokens did this output token attend to?"

We then map input token ranges to chunks to get chunk-level attribution.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from vectorlens.types import RetrievedChunk, TokenHeatmapEntry

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class AttentionAttributor:
    """Token-level attribution via attention rollout for local HuggingFace models."""

    def compute(
        self,
        model: Any,
        tokenizer: Any,
        input_text: str,
        output_text: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """
        Compute attribution scores using attention rollout.

        Algorithm:
        1. Tokenize input_text
        2. Run model with output_attentions=True
        3. Apply attention rollout across all layers:
           - Start with identity matrix
           - For each layer: rollout = rollout @ (0.5*attention + 0.5*identity)
           - This propagates attention through layers
        4. For each chunk, find its token span in the input
        5. Sum attention from last output token to that chunk's token span
        6. Normalize scores across chunks
        7. Set chunk.attribution_score

        Args:
            model: HuggingFace AutoModelForCausalLM
            tokenizer: HuggingFace tokenizer
            input_text: Full prompt text
            output_text: Model output text (used for validation)
            chunks: List of retrieved chunks to attribute

        Returns:
            Chunks with attribution_score set
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "Install torch to use attention attribution: pip install torch"
            )

        try:
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"]
            seq_len = input_ids.shape[1]

            # Get device from model
            device = model.device

            # Move inputs to model device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run model with attention output
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)

            # Check if attentions are available
            if not hasattr(outputs, "attentions") or outputs.attentions is None:
                logger.warning(
                    "Model does not support output_attentions. "
                    "Returning chunks with attribution_score=0"
                )
                for chunk in chunks:
                    chunk.attribution_score = 0.0
                return chunks

            attentions = outputs.attentions

            # Attention rollout: propagate attention through layers
            rollout = torch.eye(seq_len, device=device)

            for layer_attn in attentions:
                # layer_attn shape: (batch, heads, seq_len, seq_len)
                # Average over heads
                attn = layer_attn[0].mean(dim=0)  # (seq_len, seq_len)

                # Mix with identity for smoothing
                attn = 0.5 * attn + 0.5 * torch.eye(seq_len, device=device)

                # Normalize rows — clamp denominator to avoid zero-division NaN
                row_sums = attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn = attn / row_sums

                # Update rollout
                rollout = rollout @ attn

            # Get char-to-token mapping
            char_to_token = self._get_char_to_token_mapping(tokenizer, input_text)

            # Compute attribution for each chunk
            chunk_scores = []

            for chunk in chunks:
                # Find chunk text in input_text
                chunk_start_char = input_text.find(chunk.text[:50])

                if chunk_start_char == -1:
                    # Chunk not found in input, set score to 0
                    chunk.attribution_score = 0.0
                    chunk_scores.append(0.0)
                    continue

                # Find end of chunk
                chunk_end_char = chunk_start_char + len(chunk.text)

                # Map char offsets to token indices
                if (
                    chunk_start_char not in char_to_token
                    or chunk_end_char - 1 not in char_to_token
                ):
                    chunk.attribution_score = 0.0
                    chunk_scores.append(0.0)
                    continue

                token_start = char_to_token[chunk_start_char]
                token_end = char_to_token[chunk_end_char - 1] + 1

                # Sum attention from last output token to chunk tokens
                # rollout[-1] is attention from last token to all input tokens
                score = rollout[-1, token_start:token_end].sum().item()
                chunk_scores.append(score)

            # Normalize scores
            total_score = sum(chunk_scores)
            if total_score > 0:
                for chunk, score in zip(chunks, chunk_scores):
                    chunk.attribution_score = float(score / total_score)
            else:
                # All scores are 0, set equal weights
                for chunk in chunks:
                    chunk.attribution_score = 1.0 / len(chunks) if chunks else 0.0

            return chunks

        except Exception as e:
            # Log error and return chunks with score 0
            logger.warning(f"Error computing attention attribution: {e}")
            for chunk in chunks:
                chunk.attribution_score = 0.0
            return chunks

    def compute_per_token(
        self,
        model: Any,
        tokenizer: Any,
        prompt_text: str,
        output_text: str,
        chunks: list[RetrievedChunk],
    ) -> list[TokenHeatmapEntry]:
        """Compute per-output-subword-token chunk attribution via attention rollout.

        Unlike compute(), which returns one score per chunk using only the last
        output token's attention, this method runs the full [prompt + output]
        sequence through the model and returns one TokenHeatmapEntry per output
        subword token, each with a chunk_attributions dict.

        Algorithm:
        1. Tokenize prompt and output separately to find the prompt/output boundary
        2. Concatenate and run model with output_attentions=True
        3. Compute attention rollout over the full sequence
        4. For each output position p: extract rollout[p, 0:prompt_len] — attention
           from output token p to each prompt token — and map spans to chunks
        5. Normalize chunk scores per output token

        Returns:
            List of TokenHeatmapEntry (one per output subword), or [] on failure.
        """
        try:
            import torch
        except ImportError:
            return []

        try:
            # Tokenize prompt and output separately to know the boundary.
            # add_special_tokens=False for output so we don't double-count BOS.
            prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
            output_enc = tokenizer(output_text, return_tensors="pt", add_special_tokens=False)

            prompt_ids = prompt_enc["input_ids"]   # (1, P)
            output_ids = output_enc["input_ids"]   # (1, O)
            prompt_len = prompt_ids.shape[1]
            output_len = output_ids.shape[1]

            if output_len == 0:
                return []

            # Concatenate into one sequence
            full_ids = torch.cat([prompt_ids, output_ids], dim=1)
            full_len = full_ids.shape[1]

            device = model.device
            full_ids = full_ids.to(device)

            with torch.no_grad():
                outputs = model(full_ids, output_attentions=True)

            if not hasattr(outputs, "attentions") or outputs.attentions is None:
                logger.warning("Model does not support output_attentions; skipping token heatmap")
                return []

            # Attention rollout over full sequence
            rollout = torch.eye(full_len, device=device)
            for layer_attn in outputs.attentions:
                attn = layer_attn[0].mean(dim=0)  # (full_len, full_len)
                attn = 0.5 * attn + 0.5 * torch.eye(full_len, device=device)
                row_sums = attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn = attn / row_sums
                rollout = rollout @ attn

            # Build char→prompt-token mapping (only for prompt text)
            char_to_token = self._get_char_to_token_mapping(tokenizer, prompt_text)

            # Precompute chunk spans in prompt token space
            chunk_token_spans: list[tuple[str, int, int] | None] = []
            for chunk in chunks:
                start_char = prompt_text.find(chunk.text[:50])
                if start_char == -1:
                    chunk_token_spans.append(None)
                    continue
                end_char = start_char + len(chunk.text)
                ts = char_to_token.get(start_char)
                te = char_to_token.get(end_char - 1)
                if ts is None or te is None:
                    chunk_token_spans.append(None)
                else:
                    chunk_token_spans.append((chunk.chunk_id, ts, te + 1))

            # Decode output subword token strings for display
            output_token_ids_list = output_ids[0].tolist()
            output_token_texts = tokenizer.convert_ids_to_tokens(output_token_ids_list)

            entries: list[TokenHeatmapEntry] = []
            for i, tok_text in enumerate(output_token_texts):
                pos = prompt_len + i  # position in full_ids sequence

                # attention from this output token to all prompt positions
                attn_to_prompt = rollout[pos, :prompt_len]  # (prompt_len,)

                raw_scores: dict[str, float] = {}
                for chunk, span in zip(chunks, chunk_token_spans):
                    if span is None:
                        raw_scores[chunk.chunk_id] = 0.0
                    else:
                        _, ts, te = span
                        raw_scores[chunk.chunk_id] = attn_to_prompt[ts:te].sum().item()

                total = sum(raw_scores.values())
                if total > 0:
                    norm_scores = {k: v / total for k, v in raw_scores.items()}
                else:
                    n = len(chunks)
                    norm_scores = {k: 1.0 / n if n else 0.0 for k in raw_scores}

                # Decode subword to display text (strip sentencepiece/BPE markers)
                display = (tok_text or "")
                for marker in ("Ġ", "▁", "##"):
                    display = display.replace(marker, " " if marker != "##" else "")

                entries.append(TokenHeatmapEntry(
                    text=display,
                    position=i,
                    chunk_attributions=norm_scores,
                ))

            return entries

        except Exception as e:
            logger.warning(f"compute_per_token failed: {e}")
            return []

    def _get_char_to_token_mapping(
        self, tokenizer: Any, text: str
    ) -> dict[int, int]:
        """
        Create a mapping from character positions to token indices.

        Args:
            tokenizer: HuggingFace tokenizer
            text: Input text

        Returns:
            Dictionary: char_index -> token_index
        """
        try:
            # Use tokenizer with offset mapping
            encoded = tokenizer(
                text,
                return_offsets_mapping=True,
                return_tensors=None,
            )

            char_to_token = {}
            offsets = encoded.get("offset_mapping", [])

            for token_idx, (start, end) in enumerate(offsets):
                for char_idx in range(start, end):
                    char_to_token[char_idx] = token_idx

            return char_to_token

        except Exception:
            # Fallback: simple character-to-token mapping
            # This is a rough approximation
            encoded = tokenizer(text, return_tensors=None)
            input_ids = encoded["input_ids"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)

            char_to_token = {}
            char_idx = 0

            for token_idx, token in enumerate(tokens):
                # Estimate token length
                token_str = token.replace("Ġ", " ").replace("##", "")
                for _ in range(len(token_str)):
                    if char_idx < len(text):
                        char_to_token[char_idx] = token_idx
                        char_idx += 1

            return char_to_token
