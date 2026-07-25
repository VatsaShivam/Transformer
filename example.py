"""
End-to-end example: build a small GPT-style causal Transformer, train it
on dummy data, validate, checkpoint, and run inference with several
decoding strategies.

Run with:
    python example.py
"""

import torch
from torch.utils.data import random_split

from config import TransformerConfig
from model import TransformerModel
from data import ToyLanguageModelingDataset, build_dataloader
from train import Trainer, TrainingConfig
from inference import generate


def main() -> None:
    torch.manual_seed(42)

    # 1. Configuration -------------------------------------------------
    model_config = TransformerConfig(
        vocab_size=1000,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        max_seq_len=128,
        activation="gelu",
        prenorm=True,
        position_encoding="sinusoidal",
        causal=True,  # GPT-style autoregressive model
        pad_token_id=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    training_config = TrainingConfig(
        lr=3e-4,
        num_epochs=2,
        grad_accum_steps=2,
        max_grad_norm=1.0,
        warmup_steps=20,
        use_amp=True,
        early_stopping_patience=2,
        checkpoint_dir="./checkpoints",
        pad_token_id=model_config.pad_token_id,
    )

    # 2. Model -----------------------------------------------------------
    model = TransformerModel(model_config)
    print(f"Model parameters: {model.num_parameters():,}")

    # 3. Data --------------------------------------------------------------
    dataset = ToyLanguageModelingDataset(
        num_examples=500, vocab_size=model_config.vocab_size, min_len=8, max_len=64
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = build_dataloader(
        train_set, batch_size=16, pad_token_id=model_config.pad_token_id, shuffle=True
    )
    val_loader = build_dataloader(
        val_set, batch_size=16, pad_token_id=model_config.pad_token_id, shuffle=False
    )

    # 4. Train ---------------------------------------------------------
    trainer = Trainer(model, train_loader, val_loader, training_config)
    trainer.fit()

    # 5. Inference -------------------------------------------------------
    model.eval()
    prompt = torch.randint(1, model_config.vocab_size, (1, 5)).to(trainer.device)

    greedy_out = generate(model, prompt, max_new_tokens=10)
    print("Greedy decoding:      ", greedy_out.tolist())

    topk_out = generate(model, prompt, max_new_tokens=10, temperature=0.8, top_k=50)
    print("Top-k sampling:        ", topk_out.tolist())

    topp_out = generate(model, prompt, max_new_tokens=10, temperature=0.8, top_p=0.9)
    print("Top-p (nucleus):       ", topp_out.tolist())

    combined_out = generate(
        model, prompt, max_new_tokens=10, temperature=0.7, top_k=50, top_p=0.9
    )
    print("Top-k + top-p combined:", combined_out.tolist())


if __name__ == "__main__":
    main()
