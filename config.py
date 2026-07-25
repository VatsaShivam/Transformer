"""
Configuration for the Transformer model.

Centralizing every hyperparameter in a single dataclass keeps model
construction reproducible and makes it trivial to serialize/deserialize
experiment configs (e.g. to JSON) for checkpointing.
"""

from dataclasses import dataclass, asdict
import json
import torch


@dataclass
class TransformerConfig:
    """Hyperparameters and architectural switches for TransformerModel.

    Attributes:
        vocab_size: Size of the token vocabulary.
        embed_dim: Dimensionality of token/positional embeddings (a.k.a. d_model).
        num_heads: Number of attention heads. Must evenly divide embed_dim.
        num_layers: Number of stacked TransformerBlock layers.
        dropout: Dropout probability applied throughout the model.
        max_seq_len: Maximum sequence length supported by positional encodings.
        activation: Feed-forward activation, one of {"gelu", "relu", "silu"}.
        prenorm: If True, use Pre-LayerNorm blocks; otherwise Post-LayerNorm.
        position_encoding: Positional encoding scheme, "learned" or "sinusoidal".
        ffn_expansion: Expansion ratio for the feed-forward hidden dimension
            (hidden_dim = ffn_expansion * embed_dim).
        causal: If True, the model applies a causal (autoregressive) mask by
            default when no explicit attention_mask is supplied.
        pad_token_id: Token id used for padding (used to auto-build padding masks).
        device: Torch device string, e.g. "cuda" or "cpu".
    """

    vocab_size: int = 32000
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 1024
    activation: str = "gelu"
    prenorm: bool = True
    position_encoding: str = "learned"  # "learned" | "sinusoidal"
    ffn_expansion: int = 4
    causal: bool = False
    pad_token_id: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.position_encoding not in ("learned", "sinusoidal"):
            raise ValueError("position_encoding must be 'learned' or 'sinusoidal'")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValueError("activation must be 'gelu', 'relu', or 'silu'")

    def to_json(self, path: str) -> None:
        """Serialize the config to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TransformerConfig":
        """Load a config from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
