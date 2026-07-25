"""
Core building blocks for the Transformer model.

Each component (TokenEmbedding, PositionalEncoding, MultiHeadAttention,
FeedForward, TransformerBlock) is implemented as an independent, swappable
nn.Module so that alternative implementations (e.g. FlashAttention,
rotary embeddings, a different FFN) can be dropped in without touching
the rest of the model.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# `scaled_dot_product_attention` was added in PyTorch 2.0. We detect its
# availability once at import time and fall back to a manual implementation
# if it is missing (e.g. on older PyTorch versions).
_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")


def get_activation(name: str) -> nn.Module:
    """Return an activation module by name.

    Args:
        name: One of "gelu", "relu", "silu".

    Returns:
        An instantiated activation module.
    """
    activations = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[name]()


class TokenEmbedding(nn.Module):
    """Maps token ids to dense embeddings, scaled by sqrt(embed_dim).

    The sqrt(embed_dim) scaling follows the original "Attention Is All You
    Need" convention so that embedding magnitudes are comparable to the
    positional encoding magnitudes before they are summed.
    """

    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: LongTensor of shape (batch, seq_len).

        Returns:
            FloatTensor of shape (batch, seq_len, embed_dim).
        """
        x = self.embedding(token_ids) * self.scale
        return self.dropout(x)


class PositionalEncoding(nn.Module):
    """Adds positional information to token embeddings.

    Supports two modes, selected at construction time:
      - "learned": a trainable nn.Embedding over position indices.
      - "sinusoidal": the fixed sine/cosine scheme from Vaswani et al.,
        precomputed once and registered as a (non-trainable) buffer.
    """

    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int,
        mode: str = "learned",
        dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in ("learned", "sinusoidal"):
            raise ValueError("mode must be 'learned' or 'sinusoidal'")
        self.mode = mode
        self.dropout = nn.Dropout(dropout)

        if mode == "learned":
            self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        else:
            pe = self._build_sinusoidal_table(max_seq_len, embed_dim)
            # Registered as a buffer: moves with .to(device) but is not
            # trained and is excluded from state_dict optimizer updates.
            self.register_buffer("pe", pe, persistent=False)

    @staticmethod
    def _build_sinusoidal_table(max_seq_len: int, embed_dim: int) -> torch.Tensor:
        """Precompute the sinusoidal positional encoding table.

        pe[pos, 2i]   = sin(pos / 10000^(2i/embed_dim))
        pe[pos, 2i+1] = cos(pos / 10000^(2i/embed_dim))
        """
        position = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_seq_len, embed_dim, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_seq_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings of shape (batch, seq_len, embed_dim).

        Returns:
            x + positional encoding, same shape as input.
        """
        seq_len = x.size(1)
        if self.mode == "learned":
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            pos_emb = self.pos_embedding(positions)
        else:
            pos_emb = self.pe[:, :seq_len, :].to(x.dtype)
        return self.dropout(x + pos_emb)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional padding/causal masking.

    Uses `torch.nn.functional.scaled_dot_product_attention` (fused,
    memory-efficient, and FlashAttention-backed on supported hardware/
    PyTorch builds) when available, and transparently falls back to a
    manual implementation otherwise. Both paths are numerically
    equivalent and produce identical masking semantics.

    To swap in a custom attention kernel (e.g. a specific FlashAttention
    package), replace `_sdpa_attention` / `_manual_attention` with your
    own implementation — the surrounding module interface stays the same.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV projection: one matmul instead of three.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout_p = dropout
        self.attn_dropout = nn.Dropout(dropout)  # used only in manual fallback
        self.out_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
        """(batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, head_dim)."""
        x = x.view(batch, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    @staticmethod
    def _merge_heads(x: torch.Tensor, batch: int, seq_len: int, embed_dim: int) -> torch.Tensor:
        """(batch, num_heads, seq_len, head_dim) -> (batch, seq_len, embed_dim)."""
        return x.transpose(1, 2).contiguous().view(batch, seq_len, embed_dim)

    def _sdpa_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        attn_mask: Optional[torch.Tensor], is_causal: bool,
    ) -> torch.Tensor:
        """Fused attention path via F.scaled_dot_product_attention.

        `attn_mask` here is expected as an additive float mask (0 for
        keep, -inf for masked) or a boolean mask (True = keep), already
        broadcastable to (batch, num_heads, q_len, k_len).
        """
        dropout_p = self.attn_dropout_p if self.training else 0.0
        return F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def _manual_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        attn_mask: Optional[torch.Tensor], is_causal: bool,
    ) -> torch.Tensor:
        """Manual fallback attention (used when SDPA is unavailable)."""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (b, h, q_len, k_len)

        if is_causal:
            q_len, k_len = scores.size(-2), scores.size(-1)
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # True means "keep"; invert to fill where False (masked out).
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask  # additive mask (already -inf where masked)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, embed_dim).
            attn_mask: Optional mask broadcastable to
                (batch, 1 or num_heads, q_len, k_len). May be boolean
                (True = attend, False = mask out) or additive float
                (0 = attend, -inf = mask out).
            is_causal: If True, apply a causal (lower-triangular) mask.
                Combine with attn_mask to support causal + padding jointly.

        Returns:
            Attention output of shape (batch, seq_len, embed_dim).
        """
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)  # (b, seq_len, 3 * embed_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        q = self._split_heads(q, batch, seq_len)
        k = self._split_heads(k, batch, seq_len)
        v = self._split_heads(v, batch, seq_len)

        if _HAS_SDPA:
            attn_out = self._sdpa_attention(q, k, v, attn_mask, is_causal)
        else:
            attn_out = self._manual_attention(q, k, v, attn_mask, is_causal)

        attn_out = self._merge_heads(attn_out, batch, seq_len, self.embed_dim)
        out = self.out_proj(attn_out)
        return self.out_dropout(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> Activation -> Dropout -> Linear.

    The hidden dimension defaults to `expansion * embed_dim` (4x, following
    the original Transformer), and is fully configurable.
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = get_activation(activation)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, embed_dim).

        Returns:
            Output of shape (batch, seq_len, embed_dim).
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.dropout2(x)


class TransformerBlock(nn.Module):
    """A single Transformer encoder block with self-attention and FFN.

    Supports both Pre-LayerNorm (`prenorm=True`, generally more stable
    for training deep stacks without careful warmup) and Post-LayerNorm
    (`prenorm=False`, the original "Attention Is All You Need" ordering).

    Pre-LN:  x = x + Attn(LN(x));  x = x + FFN(LN(x))
    Post-LN: x = LN(x + Attn(x));  x = LN(x + FFN(x))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        activation: str = "gelu",
        dropout: float = 0.1,
        prenorm: bool = True,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(embed_dim, ffn_expansion, activation, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # Separate residual dropouts, applied after sub-layer output is
        # added back to the residual stream (in addition to internal
        # dropout already applied inside attn/ffn output projections).
        self.resid_dropout1 = nn.Dropout(dropout)
        self.resid_dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: Input of shape (batch, seq_len, embed_dim).
            attn_mask: Optional attention mask, see MultiHeadAttention.forward.
            is_causal: If True, apply causal masking.

        Returns:
            Output of shape (batch, seq_len, embed_dim).
        """
        if self.prenorm:
            attn_out = self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)
            x = x + self.resid_dropout1(attn_out)
            ffn_out = self.ffn(self.norm2(x))
            x = x + self.resid_dropout2(ffn_out)
        else:
            attn_out = self.attn(x, attn_mask=attn_mask, is_causal=is_causal)
            x = self.norm1(x + self.resid_dropout1(attn_out))
            ffn_out = self.ffn(x)
            x = self.norm2(x + self.resid_dropout2(ffn_out))
        return x


def initialize_weights(module: nn.Module) -> None:
    """Apply project-standard weight initialization to a module (in place).

    - Linear layers: Xavier (Glorot) uniform init for weights, zeros for bias.
    - Embedding layers: normal init (mean=0, std=0.02), matching common
      GPT/BERT-style initialization.
    - LayerNorm: weight=1, bias=0 (the identity transform at init).

    Typical usage: `model.apply(initialize_weights)`.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
