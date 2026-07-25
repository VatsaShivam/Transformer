"""
Training utilities for TransformerModel.

Includes a `Trainer` class that wraps:
  - CrossEntropyLoss (with padding ignored via ignore_index)
  - AdamW optimizer
  - Linear-warmup / cosine-decay LR scheduler
  - Gradient clipping
  - Mixed precision training (torch.cuda.amp)
  - Gradient accumulation
  - Checkpoint saving / loading
  - Early stopping on validation loss
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """Linear warmup followed by cosine decay to zero.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        num_warmup_steps: Number of steps to linearly ramp LR from 0 -> base LR.
        num_training_steps: Total number of training steps (for cosine decay span).

    Returns:
        A LambdaLR scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@dataclass
class TrainingConfig:
    """Hyperparameters controlling the training loop.

    Attributes:
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        num_epochs: Number of epochs to train for.
        grad_accum_steps: Number of micro-batches to accumulate before an
            optimizer step (effective batch size = batch_size * this).
        max_grad_norm: Gradient clipping threshold (L2 norm).
        warmup_steps: Number of LR warmup steps.
        use_amp: Whether to use automatic mixed precision (requires CUDA
            for real speedups; falls back to a no-op on CPU).
        early_stopping_patience: Stop training if validation loss does not
            improve for this many consecutive validation checks.
        checkpoint_dir: Directory to save checkpoints to.
        pad_token_id: Padding id, passed to the loss as ignore_index.
    """

    lr: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 3
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    use_amp: bool = True
    early_stopping_patience: int = 3
    checkpoint_dir: str = "./checkpoints"
    pad_token_id: int = 0


class EarlyStopping:
    """Tracks validation loss and signals when training should stop."""

    def __init__(self, patience: int = 3, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        Returns:
            True if training should stop.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Encapsulates the training loop for TransformerModel.

    Example:
        trainer = Trainer(model, train_loader, val_loader, training_config)
        trainer.fit()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: TrainingConfig,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        self.optimizer = AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        total_steps = (
            len(train_loader) // max(1, config.grad_accum_steps) * config.num_epochs
        )
        self.scheduler = build_warmup_cosine_scheduler(
            self.optimizer, config.warmup_steps, max(1, total_steps)
        )

        # GradScaler only does meaningful work on CUDA; on CPU it is a
        # harmless no-op wrapper, keeping the code path unified.
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and self.device == "cuda")
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        self.global_step = 0

    def _forward_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run a forward pass and compute the cross-entropy loss for one batch."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        logits = self.model(input_ids, attention_mask=attention_mask)
        # (batch, seq_len, vocab) -> (batch * seq_len, vocab) for CE loss.
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
        )
        return loss

    def train_epoch(self) -> float:
        """Run one epoch of training. Returns average training loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        amp_enabled = self.config.use_amp and self.device == "cuda"

        self.optimizer.zero_grad()
        for step, batch in enumerate(self.train_loader):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = self._forward_loss(batch)
                loss_to_backprop = loss / self.config.grad_accum_steps

            self.scaler.scale(loss_to_backprop).backward()

            if (step + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            total_loss += loss.item()
            num_batches += 1

        # Flush any leftover accumulated gradients if the number of
        # batches in the epoch wasn't evenly divisible by
        # grad_accum_steps (otherwise the last partial accumulation
        # window would silently be dropped).
        if num_batches % self.config.grad_accum_steps != 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            self.global_step += 1

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self) -> float:
        """Run validation. Returns average validation loss."""
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_loader:
            loss = self._forward_loss(batch)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / max(1, num_batches)

    def save_checkpoint(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Save model, optimizer, scheduler, and scaler state to disk."""
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load model, optimizer, scheduler, and scaler state from disk."""
        state = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.scaler.load_state_dict(state["scaler_state_dict"])
        self.global_step = state.get("global_step", 0)
        return state

    def fit(self) -> None:
        """Run the full training loop with early stopping and checkpointing."""
        best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.evaluate()
            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs} "
                f"- train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}"
            )

            if self.val_loader is not None:
                if val_loss < self.early_stopping.best_loss:
                    self.save_checkpoint(best_path, extra={"epoch": epoch, "val_loss": val_loss})
                if self.early_stopping.step(val_loss):
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break
