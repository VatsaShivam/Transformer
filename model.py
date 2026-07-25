"""
Top-level Transformer model.

Assembles TokenEmbedding + PositionalEncoding + a stack of
TransformerBlocks + a final projection to vocabulary logits. Designed to
work as either a BERT-style encoder (causal=False, use padding masks) or
a GPT-style decoder (causal=True) depending on configuration and the
masks passed to `forward`.
"""

from typing import Optional

import torch
import torch.nn as nn

from config import TransformerConfig
from modules import TokenEmbedding, PositionalEncoding, TransformerBlock, initialize_weights


def build_padding_mask(
    token_ids: torch.Tensor, pad_token_id: int, num_heads: int
) -> torch.Tensor:
    """Build a boolean attention mask from padding token positions.

    Args:
        token_ids: LongTensor of shape (batch, seq_len).
        pad_token_id: The id representing padding.
        num_heads: Number of attention heads (mask is broadcast, so this
            is only used for documentation of the expected shape; a
            singleton head dim broadcasts automatically).

    Returns:
        BoolTensor of shape (batch, 1, 1, seq_len) where True means
        "attend to this key position" and False means "mask out". This
        shape broadcasts across the query dimension and all heads.
    """
    # (batch, seq_len) -> (batch, 1, 1, seq_len)
    key_padding = token_ids != pad_token_id
    return key_padding[:, None, None, :]


def combine_masks(
    padding_mask: Optional[torch.Tensor], seq_len: int, causal: bool, device: torch.device
) -> Optional[torch.Tensor]:
    """Combine an optional padding mask with an optional causal mask.

    Args:
        padding_mask: BoolTensor (batch, 1, 1, k_len) or None.
        seq_len: Query/key sequence length (assumed equal, self-attention).
        causal: Whether to additionally apply a causal (lower-triangular) mask.
        device: Device to build the causal mask on.

    Returns:
        A boolean mask broadcastable to (batch, heads, seq_len, seq_len)
        with True = attend, False = mask out. None if no masking is needed.
    """
    mask = padding_mask
    if causal:
        causal_bool = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )[None, None, :, :]  # (1, 1, q_len, k_len)
        mask = causal_bool if mask is None else (mask & causal_bool)
    return mask


class TransformerModel(nn.Module):
    """A configurable, GPT/BERT-style Transformer for sequence modeling.

    The model can operate as:
      - An encoder (bidirectional attention) when `config.causal=False`.
      - An autoregressive decoder (causal attention) when
        `config.causal=True`, suitable for next-token prediction / GPT-style
        generation.

    Masking:
      `forward` accepts an explicit `attention_mask` (1 = real token,
      0 = padding, shape (batch, seq_len)) and/or relies on
      `config.causal` to build a causal mask automatically. Both can be
      combined for causal decoding over padded batches.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(
            config.vocab_size, config.embed_dim, dropout=config.dropout
        )
        self.positional_encoding = PositionalEncoding(
            config.embed_dim,
            config.max_seq_len,
            mode=config.position_encoding,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ffn_expansion=config.ffn_expansion,
                    activation=config.activation,
                    dropout=config.dropout,
                    prenorm=config.prenorm,
                )
                for _ in range(config.num_layers)
            ]
        )
        # Final norm is standard practice for Pre-LN stacks (stabilizes the
        # output distribution before the projection head); harmless for
        # Post-LN too.
        self.final_norm = nn.LayerNorm(config.embed_dim)
        self.output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Initialize BEFORE tying weights: initialize_weights() would
        # otherwise visit both the embedding (normal_ init) and the linear
        # output projection (xavier_uniform_ init) with different rules
        # while they share the same underlying tensor, making the final
        # values depend on module traversal order. Initializing first and
        # tying second guarantees the final, deterministic values come
        # from the embedding's initialization scheme.
        self.apply(initialize_weights)

        # Weight tying between the input embedding and output projection is
        # a common, effective trick (fewer parameters, often better
        # perplexity). Enabled by default here.
        self.output_proj.weight = self.token_embedding.embedding.weight

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: LongTensor of shape (batch, seq_len).
            attention_mask: Optional tensor of shape (batch, seq_len) with
                1 for real tokens and 0 for padding. If None, no padding
                mask is applied (assumes no padding, or padding handled
                upstream).

        Returns:
            Logits over the vocabulary, shape (batch, seq_len, vocab_size).
        """
        batch, seq_len = token_ids.shape
        device = token_ids.device

        padding_mask = None
        if attention_mask is not None:
            # (batch, seq_len) -> (batch, 1, 1, seq_len), True = attend.
            padding_mask = attention_mask.bool()[:, None, None, :]

        mask = combine_masks(padding_mask, seq_len, self.config.causal, device)

        x = self.token_embedding(token_ids)
        x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(x, attn_mask=mask, is_causal=False)
            # Note: causal-ness is already baked into `mask` via
            # combine_masks, so we pass is_causal=False here and let the
            # explicit boolean mask do the work. This keeps the mask
            # correct even when padding and causality must both apply
            # (SDPA's is_causal=True flag cannot be combined with a
            # separate attn_mask in older PyTorch versions).

        x = self.final_norm(x)
        logits = self.output_proj(x)
        return logits

    @torch.no_grad()
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters (useful for logging/model cards)."""
        params = (p for p in self.parameters() if not trainable_only or p.requires_grad)
        return sum(p.numel() for p in params)
