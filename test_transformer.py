"""
Unit tests for the Transformer implementation.

Run with:
    python -m pytest test_transformer.py -v
or:
    python test_transformer.py
"""

import math
import unittest

import torch

from config import TransformerConfig
from model import TransformerModel, build_padding_mask, combine_masks
from modules import PositionalEncoding, MultiHeadAttention


class TestPositionalEncoding(unittest.TestCase):
    def test_sinusoidal_values_correct(self):
        """Spot-check the sinusoidal formula against a hand computation."""
        embed_dim, max_len = 16, 10
        pe_module = PositionalEncoding(embed_dim, max_len, mode="sinusoidal", dropout=0.0)
        table = pe_module.pe[0]  # (max_len, embed_dim)

        pos, dim_pair = 3, 2  # check position=3, feature indices 4/5
        expected_angle = pos / (10000 ** (dim_pair * 2 / embed_dim))
        self.assertAlmostEqual(
            table[pos, dim_pair * 2].item(), math.sin(expected_angle), places=4
        )
        self.assertAlmostEqual(
            table[pos, dim_pair * 2 + 1].item(), math.cos(expected_angle), places=4
        )

    def test_learned_positional_shapes(self):
        embed_dim, max_len, batch, seq_len = 32, 50, 4, 12
        pe_module = PositionalEncoding(embed_dim, max_len, mode="learned", dropout=0.0)
        x = torch.zeros(batch, seq_len, embed_dim)
        out = pe_module(x)
        self.assertEqual(out.shape, (batch, seq_len, embed_dim))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            PositionalEncoding(16, 10, mode="invalid")


class TestMasking(unittest.TestCase):
    def test_padding_mask_shape_and_values(self):
        token_ids = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
        mask = build_padding_mask(token_ids, pad_token_id=0, num_heads=4)
        self.assertEqual(mask.shape, (2, 1, 1, 4))
        expected = torch.tensor(
            [[[[True, True, False, False]]], [[[True, True, True, False]]]]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_causal_mask_is_lower_triangular(self):
        seq_len = 5
        mask = combine_masks(None, seq_len, causal=True, device=torch.device("cpu"))
        expected = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        self.assertTrue(torch.equal(mask[0, 0], expected))

    def test_combined_padding_and_causal_mask(self):
        token_ids = torch.tensor([[1, 1, 0]])  # last position padded
        padding_mask = build_padding_mask(token_ids, pad_token_id=0, num_heads=1)
        mask = combine_masks(padding_mask, seq_len=3, causal=True, device=torch.device("cpu"))
        # Query 0 can only see key 0 (causal), and key 0 is not padded -> True.
        self.assertTrue(mask[0, 0, 0, 0].item())
        # Query 2 (last) could see keys 0,1,2 causally, but key 2 is padding -> False.
        self.assertFalse(mask[0, 0, 2, 2].item())
        self.assertTrue(mask[0, 0, 2, 0].item())


class TestMultiHeadAttention(unittest.TestCase):
    def test_output_shape(self):
        batch, seq_len, embed_dim, num_heads = 3, 7, 32, 4
        attn = MultiHeadAttention(embed_dim, num_heads, dropout=0.0)
        x = torch.randn(batch, seq_len, embed_dim)
        out = attn(x)
        self.assertEqual(out.shape, (batch, seq_len, embed_dim))

    def test_causal_attention_does_not_leak_future(self):
        """A change to a future token must not affect an earlier position's
        output under causal masking."""
        torch.manual_seed(0)
        batch, seq_len, embed_dim, num_heads = 1, 5, 16, 2
        attn = MultiHeadAttention(embed_dim, num_heads, dropout=0.0)
        attn.eval()

        x = torch.randn(batch, seq_len, embed_dim)
        out1 = attn(x, is_causal=True)

        x_modified = x.clone()
        x_modified[:, -1, :] += 100.0  # perturb only the last (future) token
        out2 = attn(x_modified, is_causal=True)

        # All positions except the last should be unaffected by the change
        # to the last token, since causal masking prevents looking ahead.
        self.assertTrue(torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5))
        # The last position's output SHOULD differ (it attends to itself).
        self.assertFalse(torch.allclose(out1[:, -1, :], out2[:, -1, :], atol=1e-5))


class TestTransformerModel(unittest.TestCase):
    def _small_config(self, **overrides) -> TransformerConfig:
        defaults = dict(
            vocab_size=100,
            embed_dim=32,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            max_seq_len=64,
            activation="gelu",
            prenorm=True,
            position_encoding="learned",
            causal=False,
            device="cpu",
        )
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def test_forward_shape(self):
        config = self._small_config()
        model = TransformerModel(config)
        batch, seq_len = 4, 10
        token_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        logits = model(token_ids)
        self.assertEqual(logits.shape, (batch, seq_len, config.vocab_size))

    def test_forward_with_padding_mask(self):
        config = self._small_config()
        model = TransformerModel(config)
        token_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 0]])
        attention_mask = (token_ids != 0).long()
        logits = model(token_ids, attention_mask=attention_mask)
        self.assertEqual(logits.shape, (2, 5, config.vocab_size))
        self.assertFalse(torch.isnan(logits).any())

    def test_backward_pass_produces_gradients(self):
        config = self._small_config()
        model = TransformerModel(config)
        token_ids = torch.randint(0, config.vocab_size, (2, 8))
        logits = model(token_ids)
        loss = logits.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and torch.any(p.grad != 0) for p in model.parameters()
        )
        self.assertTrue(has_grad)

    def test_causal_model_next_token_independent_of_future(self):
        config = self._small_config(causal=True)
        model = TransformerModel(config)
        model.eval()

        token_ids = torch.randint(1, config.vocab_size, (1, 6))
        logits1 = model(token_ids)

        token_ids_modified = token_ids.clone()
        token_ids_modified[0, -1] = (token_ids_modified[0, -1] + 1) % config.vocab_size
        logits2 = model(token_ids_modified)

        # Logits at all positions except the last must be unaffected.
        self.assertTrue(torch.allclose(logits1[:, :-1, :], logits2[:, :-1, :], atol=1e-4))

    def test_sinusoidal_and_learned_both_work(self):
        for mode in ("learned", "sinusoidal"):
            config = self._small_config(position_encoding=mode)
            model = TransformerModel(config)
            token_ids = torch.randint(0, config.vocab_size, (2, 5))
            logits = model(token_ids)
            self.assertEqual(logits.shape, (2, 5, config.vocab_size))

    def test_prenorm_and_postnorm_both_work(self):
        for prenorm in (True, False):
            config = self._small_config(prenorm=prenorm)
            model = TransformerModel(config)
            token_ids = torch.randint(0, config.vocab_size, (2, 5))
            logits = model(token_ids)
            self.assertEqual(logits.shape, (2, 5, config.vocab_size))

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            self._small_config(embed_dim=33, num_heads=4)  # not divisible


if __name__ == "__main__":
    unittest.main()
