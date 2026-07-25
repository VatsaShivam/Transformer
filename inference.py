"""
Inference / decoding utilities for autoregressive generation:
greedy decoding, temperature scaling, top-k sampling, and top-p
(nucleus) sampling. These can be combined (e.g. temperature + top-k
+ top-p) as in typical GPT-style sampling pipelines.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from model import TransformerModel


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature. temperature < 1 sharpens, > 1 flattens."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return logits / temperature


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out (set to -inf) all but the top-k logits per row.

    Args:
        logits: Shape (batch, vocab_size).
        k: Number of highest-probability tokens to keep.

    Returns:
        Filtered logits, same shape, with non-top-k entries set to -inf.
    """
    if k <= 0:
        return logits
    k = min(k, logits.size(-1))
    top_values, _ = torch.topk(logits, k, dim=-1)
    min_keep_value = top_values[:, -1, None]
    return torch.where(logits < min_keep_value, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) filtering: keep the smallest set of tokens whose
    cumulative probability exceeds p.

    Args:
        logits: Shape (batch, vocab_size).
        p: Cumulative probability threshold in (0, 1].

    Returns:
        Filtered logits, same shape, with tokens outside the nucleus set
        to -inf.
    """
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Shift right so we always keep at least the first (highest-prob) token.
    sorted_mask = cumulative_probs > p
    sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
    sorted_mask[:, 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))

    # Scatter back to the original (unsorted) vocabulary order.
    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits.scatter_(-1, sorted_indices, sorted_logits)
    return filtered_logits


@torch.no_grad()
def generate(
    model: TransformerModel,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Autoregressively generate tokens from a causal TransformerModel.

    Decoding strategy is determined by which sampling parameters are set:
      - top_k is None and top_p is None -> greedy decoding (argmax).
      - Otherwise -> sampling, optionally restricted by top_k and/or top_p,
        after temperature scaling.

    Args:
        model: A TransformerModel configured with `causal=True`.
        input_ids: LongTensor of shape (batch, seq_len), the prompt.
        max_new_tokens: Number of tokens to generate beyond the prompt.
        temperature: Softmax temperature (applied before top_k/top_p).
        top_k: If set, restrict sampling to the top_k most likely tokens.
        top_p: If set, restrict sampling to the smallest nucleus with
            cumulative probability >= top_p.
        eos_token_id: If provided, generation for a sequence stops (pads)
            once this token is produced.
        pad_token_id: Used to pad finished sequences within a batch.

    Returns:
        LongTensor of shape (batch, seq_len + max_new_tokens).
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.size(0)
    generated = input_ids
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    greedy = top_k is None and top_p is None

    for _ in range(max_new_tokens):
        attention_mask = (generated != pad_token_id).long()
        logits = model(generated, attention_mask=attention_mask)
        next_token_logits = logits[:, -1, :]  # (batch, vocab_size)

        if greedy:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            scaled_logits = apply_temperature(next_token_logits, temperature)
            if top_k is not None:
                scaled_logits = top_k_filter(scaled_logits, top_k)
            if top_p is not None:
                scaled_logits = top_p_filter(scaled_logits, top_p)
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        if eos_token_id is not None:
            # Once a sequence is finished, force-pad subsequent tokens.
            next_token = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(next_token, pad_token_id),
                next_token,
            )
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and finished.all():
            break

    return generated
