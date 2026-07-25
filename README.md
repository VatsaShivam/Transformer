# Modern Transformer (Educational, Production-Style)

A modular PyTorch implementation of a GPT/BERT-style Transformer, upgraded
from a basic encoder to support configurable positional encodings,
padding/causal attention masking, dropout throughout, selectable
activations, Pre/Post-LayerNorm, fused scaled-dot-product attention (with
manual fallback), a full training loop, a data pipeline, and multiple
decoding strategies for generation.

## File layout

| File                  | Contents |
|------------------------|----------|
| `config.py`            | `TransformerConfig` dataclass — all architectural/training switches. |
| `modules.py`           | `TokenEmbedding`, `PositionalEncoding`, `MultiHeadAttention`, `FeedForward`, `TransformerBlock`, `initialize_weights`. |
| `model.py`             | `TransformerModel` (assembles the pieces), `build_padding_mask`, `combine_masks`. |
| `data.py`              | `ToyLanguageModelingDataset`, dynamic-padding `collate_fn`, `build_dataloader`. |
| `train.py`             | `Trainer` — AdamW, warmup+cosine LR schedule, AMP, grad accumulation/clipping, checkpointing, early stopping. |
| `inference.py`         | `generate()` — greedy / top-k / top-p / temperature decoding. |
| `test_transformer.py`  | Unit tests: shapes, masking correctness, positional encoding math, causal leakage, forward/backward passes. |
| `example.py`           | End-to-end script: build model → train → validate → generate. |

## Quick start

```bash
pip install torch
python example.py                       # full training + generation demo
python -m pytest test_transformer.py -v # unit tests
```

## Key design choices

- **Positional encoding** (`config.position_encoding`): `"learned"` uses a
  trainable `nn.Embedding` over positions; `"sinusoidal"` precomputes the
  fixed sine/cosine table from *Attention Is All You Need* and stores it as
  a non-trainable buffer.

- **Masking**: `TransformerModel.forward` accepts an `attention_mask`
  (1 = real token, 0 = padding) and combines it with a causal mask when
  `config.causal=True`, so a single boolean mask (True = attend) is passed
  down through every block — this avoids the ambiguity of mixing SDPA's
  built-in `is_causal` flag with an explicit padding mask.

- **Attention backend**: `MultiHeadAttention` uses
  `F.scaled_dot_product_attention` when present (PyTorch ≥ 2.0; this is
  what gives you FlashAttention-backed kernels on supported GPUs) and
  transparently falls back to a manual softmax/matmul implementation
  otherwise. Swap `_sdpa_attention`/`_manual_attention` to plug in a
  custom kernel without touching anything else.

- **Pre-LN vs Post-LN** (`config.prenorm`): Pre-LN (`True`) normalizes
  before each sub-layer and tends to train more stably at depth; Post-LN
  (`False`) matches the original Transformer paper's ordering.

- **Weight tying**: the output projection shares weights with the token
  embedding (a standard trick that reduces parameters). Initialization
  runs *before* the tying assignment so the shared tensor ends up with a
  single, deterministic initialization scheme.

- **Config-driven**: every architectural knob (heads, layers, activation,
  dropout, norm placement, positional scheme, causality) lives in
  `TransformerConfig`, so experiments are just different dataclass
  instances (and trivially serializable to/from JSON).

## Note on this delivery

This sandbox has no network access, so `torch` could not be installed
here to execute the test suite. All files were syntax-verified with
`ast.parse`, and the logic was carefully hand-traced (mask broadcasting,
causal-leakage semantics, weight-tying/init order, gradient-accumulation
flushing, etc.). Please run `python -m pytest test_transformer.py -v` in
an environment with PyTorch ≥ 2.0 installed to confirm before relying on
it in production.
