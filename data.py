"""
Example data pipeline: a toy Dataset plus a collate function that performs
dynamic padding and generates attention masks on the fly.

In real usage you would replace `ToyLanguageModelingDataset` with a
dataset backed by tokenized text files, HuggingFace `datasets`, etc. The
collate function and padding strategy generalize as-is.
"""

import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader


class ToyLanguageModelingDataset(Dataset):
    """A toy dataset of random token sequences for next-token prediction.

    Each example is a random-length sequence of random token ids. Labels
    are the input sequence shifted by one position (standard causal LM
    setup): the model at position i predicts the token at position i+1.
    """

    def __init__(
        self,
        num_examples: int,
        vocab_size: int,
        min_len: int = 8,
        max_len: int = 64,
        seed: int = 0,
    ):
        self.vocab_size = vocab_size
        rng = random.Random(seed)
        self.examples: List[List[int]] = [
            [rng.randint(1, vocab_size - 1) for _ in range(rng.randint(min_len, max_len))]
            for _ in range(num_examples)
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> List[int]:
        return self.examples[idx]


def make_collate_fn(pad_token_id: int = 0):
    """Return a collate function that dynamically pads a batch of sequences.

    Args:
        pad_token_id: Token id used to pad shorter sequences.

    Returns:
        A callable suitable for `DataLoader(..., collate_fn=...)` that
        returns a dict with "input_ids", "labels", and "attention_mask",
        all shape (batch, max_len_in_batch).
    """

    def collate_fn(batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(seq) for seq in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, seq in enumerate(batch):
            length = len(seq)
            input_ids[i, :length] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, :length] = 1

        # Causal LM labels: shift input left by one; the last position has
        # no "next token" so it's set to pad (and should be excluded from
        # the loss via ignore_index).
        labels = torch.full_like(input_ids, pad_token_id)
        labels[:, :-1] = input_ids[:, 1:]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return collate_fn


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    pad_token_id: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience wrapper to build a DataLoader with dynamic padding."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=make_collate_fn(pad_token_id),
    )
