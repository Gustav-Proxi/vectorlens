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

            # Filter out None layers — modern transformers uses SDPA by default
            # which doesn't materialise attention weights. Layers using SDPA
            # return None instead of a tensor. If ALL layers are None the model
            # was loaded without eager attention; warn and fall back to zero scores.
            valid_attentions = [la for la in attentions if la is not None]
            if not valid_attentions:
                logger.warning(
                    "Model returned no attention weights (all layers None). "
                    "This usually means SDPA/flash-attention is active. "
                    "Load with attn_implementation='eager' for attention attribution."
                )
                for chunk in chunks:
                    chunk.attribution_score = 0.0
                return chunks

            # Attention rollout: propagate attention through layers
            rollout = torch.eye(seq_len, device=device)

            for layer_attn in valid_attentions:
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

            # Compute attribution for each chunk.
            # Use None as sentinel to distinguish "found but score=0" from "not found".
            chunk_scores: list[float | None] = []

            for chunk in chunks:
                # Find chunk text in input_text
                chunk_start_char = input_text.find(chunk.text[:500])

                if chunk_start_char == -1:
                    chunk.attribution_score = 0.0
                    chunk_scores.append(None)   # not found
                    continue

                # Find end of chunk
                chunk_end_char = chunk_start_char + len(chunk.text)

                # Map char offsets to token indices
                ts = char_to_token.get(chunk_start_char)
                te = char_to_token.get(chunk_end_char - 1)
                if ts is None or te is None or ts >= te:
                    chunk.attribution_score = 0.0
                    chunk_scores.append(None)   # span unmappable
                    continue

                score = rollout[-1, ts:te + 1].sum().item()
                chunk_scores.append(score)

            # Normalize only among chunks whose spans were actually found.
            # Chunks with None score (not in prompt) stay at 0.0.
            found_scores = [(i, s) for i, s in enumerate(chunk_scores) if s is not None]
            total_score = sum(s for _, s in found_scores)
            if total_score > 0:
                for i, score in found_scores:
                    chunks[i].attribution_score = float(score / total_score)
            elif found_scores:
                # Chunks were found but all rolled-out to 0 — assign equal weight
                # among found chunks only.
                eq = 1.0 / len(found_scores)
                for i, _ in found_scores:
                    chunks[i].attribution_score = eq

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
            # Both use add_special_tokens=True for prompt (BOS etc.) and False
            # for output to avoid double-counting BOS. Critically, the
            # char_to_token mapping is built with the SAME add_special_tokens=True
            # setting so token indices stay consistent with prompt_len.
            prompt_enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
            output_enc = tokenizer(output_text, return_tensors="pt", add_special_tokens=False)

            prompt_ids = prompt_enc["input_ids"]   # (1, P)  includes BOS if model uses one
            output_ids = output_enc["input_ids"]   # (1, O)
            prompt_len = prompt_ids.shape[1]
            output_len = output_ids.shape[1]

            if output_len == 0:
                return []

            # Guard against sequences that would OOM (rollout is O(seq_len^2)).
            # 4096 tokens @ float32 → ~64 MB for the rollout matrix alone.
            full_len_est = prompt_len + output_len
            if full_len_est > 4096:
                logger.warning(
                    f"Sequence too long for token heatmap ({full_len_est} tokens > 4096); "
                    "skipping to avoid OOM"
                )
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

            # Attention rollout over full sequence — skip None layers (SDPA)
            valid_attentions = [la for la in outputs.attentions if la is not None]
            if not valid_attentions:
                logger.warning(
                    "compute_per_token: no attention weights available (SDPA active). "
                    "Load model with attn_implementation='eager'."
                )
                return []

            rollout = torch.eye(full_len, device=device)
            for layer_attn in valid_attentions:
                attn = layer_attn[0].mean(dim=0)  # (full_len, full_len)
                attn = 0.5 * attn + 0.5 * torch.eye(full_len, device=device)
                row_sums = attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                attn = attn / row_sums
                rollout = rollout @ attn

            # Build char→prompt-token mapping using the SAME add_special_tokens=True
            # setting as prompt_ids so BOS/EOS offsets are consistent.
            char_to_token = self._get_char_to_token_mapping(
                tokenizer, prompt_text, add_special_tokens=True
            )

            # Precompute chunk spans in prompt token space
            chunk_token_spans: list[tuple[str, int, int] | None] = []
            for chunk in chunks:
                start_char = prompt_text.find(chunk.text[:500])
                if start_char == -1:
                    chunk_token_spans.append(None)
                    continue
                end_char = start_char + len(chunk.text)
                ts = char_to_token.get(start_char)
                te = char_to_token.get(end_char - 1)
                if ts is None or te is None or ts >= te:
                    chunk_token_spans.append(None)
                else:
                    chunk_token_spans.append((chunk.chunk_id, ts, te + 1))

            # Decode output subword token strings for display using the tokenizer's
            # own converter — handles all BPE/sentencepiece marker styles correctly.
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

                # Use tokenizer's own converter for display text — handles all
                # marker styles (Ġ for GPT-2, ▁ for SentencePiece, ## for BERT,
                # control tokens for Llama/Mistral, etc.)
                try:
                    display = tokenizer.convert_tokens_to_string([tok_text or ""])
                except Exception:
                    display = (tok_text or "")

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
        self, tokenizer: Any, text: str, add_special_tokens: "bool | None" = None
    ) -> dict[int, int]:
        """
        Create a mapping from character positions to token indices.

        Args:
            tokenizer: HuggingFace tokenizer
            text: Input text
            add_special_tokens: When provided, passed explicitly to the tokenizer.
                Must match the add_special_tokens setting used when tokenizing the
                full sequence. When None (default), the tokenizer's own default is
                used — preserves backward-compatible behaviour for compute().

        Returns:
            Dictionary: char_index -> token_index
        """
        try:
            # Use tokenizer with offset mapping
            kwargs: dict = {"return_offsets_mapping": True, "return_tensors": None}
            if add_special_tokens is not None:
                kwargs["add_special_tokens"] = add_special_tokens
            encoded = tokenizer(text, **kwargs)

            char_to_token = {}
            offsets = encoded.get("offset_mapping", [])

            for token_idx, offset in enumerate(offsets):
                # Some tokenizers return None for special tokens (BOS/EOS) that
                # have no corresponding character span — skip them.
                if offset is None:
                    continue
                start, end = offset
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
