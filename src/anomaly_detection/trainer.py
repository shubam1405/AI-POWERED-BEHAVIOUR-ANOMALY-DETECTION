"""
trainer.py – Training pipeline for the GRU Autoencoder.

Features
--------
* Trains exclusively on normal (non-anomalous) sessions.
* Masked MSE reconstruction loss (ignores zero-padded time-steps).
* Per-epoch validation loss tracking.
* Early stopping with configurable patience.
* Learning-rate scheduling (ReduceLROnPlateau).
* Best-model checkpointing (lowest validation loss).
* Resume-from-checkpoint support.
* Optional TensorBoard logging.
* Epoch-timing logs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from anomaly_detection.gru_autoencoder import GRUAutoencoder, MaskedMSELoss
from anomaly_detection.utils import elapsed_str, ensure_dir, get_device, set_seed

logger = logging.getLogger("AnomalyDetection.Trainer")


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

class TrainingResult:
    """Container for training history and final metrics."""

    def __init__(self) -> None:
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.epoch_times: List[float] = []
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.total_epochs: int = 0
        self.stopped_early: bool = False

    def as_dict(self) -> Dict:
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "epoch_times": self.epoch_times,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "total_epochs": self.total_epochs,
            "stopped_early": self.stopped_early,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GRUTrainer:
    """Training pipeline for the GRU Autoencoder.

    Parameters
    ----------
    model : GRUAutoencoder
        The model to train.
    device : torch.device, optional
        Defaults to automatic selection via :func:`get_device`.
    epochs : int
        Maximum number of training epochs (default 50).
    learning_rate : float
        Initial learning rate for AdamW (default 1e-3).
    weight_decay : float
        L2 regularisation coefficient (default 1e-5).
    patience : int
        Early-stopping patience in epochs (default 10).
    lr_patience : int
        ReduceLROnPlateau patience (default 5).
    lr_factor : float
        Factor by which to reduce LR on plateau (default 0.5).
    checkpoint_path : str
        Where to save the best model checkpoint.
    log_dir : str, optional
        If provided, TensorBoard logs are written here.
    seed : int
        Random seed for reproducibility.
    grad_clip : float
        Maximum gradient norm (default 1.0, 0 = disabled).
    """

    def __init__(
        self,
        model: GRUAutoencoder,
        device: Optional[torch.device] = None,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        lr_patience: int = 5,
        lr_factor: float = 0.5,
        checkpoint_path: str = "models/gru_autoencoder.pt",
        log_dir: Optional[str] = None,
        seed: int = 42,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.device = device or get_device()
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.checkpoint_path = checkpoint_path
        self.log_dir = log_dir
        self.seed = seed
        self.grad_clip = grad_clip

        set_seed(seed)
        self.model.to(self.device)

        self.criterion = MaskedMSELoss(reduction="mean")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=lr_factor,
            patience=lr_patience,
        )

        self._writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                ensure_dir(log_dir)
                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info("TensorBoard logging enabled at %s.", log_dir)
            except ImportError:
                logger.warning("tensorboard not installed – skipping TensorBoard logging.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> TrainingResult:
        """Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader
            Batches of normal sessions for training.
        val_loader : DataLoader
            Batches of normal sessions for validation.
        resume_from : str, optional
            Path to a checkpoint to resume from.

        Returns
        -------
        TrainingResult
        """
        result = TrainingResult()
        start_epoch = 0

        if resume_from and Path(resume_from).exists():
            start_epoch = self._load_checkpoint(resume_from, result)
            logger.info("Resuming from epoch %d.", start_epoch + 1)

        best_val_loss = result.best_val_loss
        no_improve_count = 0

        logger.info(
            "Training started  epochs=%d  lr=%.0e  device=%s",
            self.epochs, self.learning_rate, self.device,
        )

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()

            train_loss = self._run_epoch(train_loader, training=True)
            val_loss   = self._run_epoch(val_loader,   training=False)

            self.scheduler.step(val_loss)
            epoch_time = time.time() - t0

            result.train_losses.append(train_loss)
            result.val_losses.append(val_loss)
            result.epoch_times.append(epoch_time)
            result.total_epochs = epoch + 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch [%03d/%03d]  train_loss=%.6f  val_loss=%.6f  "
                "lr=%.2e  time=%s",
                epoch + 1, self.epochs,
                train_loss, val_loss,
                current_lr,
                elapsed_str(t0),
            )

            if self._writer:
                self._writer.add_scalar("Loss/train", train_loss, epoch)
                self._writer.add_scalar("Loss/val",   val_loss,   epoch)
                self._writer.add_scalar("LR",         current_lr, epoch)

            # Checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                result.best_val_loss = best_val_loss
                result.best_epoch = epoch + 1
                no_improve_count = 0
                self.model.save(self.checkpoint_path)
                logger.info("  ✓ New best val_loss=%.6f  checkpoint saved.", best_val_loss)
            else:
                no_improve_count += 1
                logger.debug("  No improvement for %d epoch(s).", no_improve_count)

            # Early stopping
            if no_improve_count >= self.patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self.patience,
                )
                result.stopped_early = True
                break

        if self._writer:
            self._writer.close()

        logger.info(
            "Training complete.  Best val_loss=%.6f at epoch %d.",
            result.best_val_loss, result.best_epoch,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        """Run one full pass over ``loader``.

        Parameters
        ----------
        loader : DataLoader
        training : bool
            If *True*, gradients are computed and the model is updated.

        Returns
        -------
        float
            Mean reconstruction loss over the epoch.
        """
        self.model.train(training)
        total_loss = 0.0
        total_batches = 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for batch in loader:
                seq, mask, _tab, _lbl, _ids = batch
                seq  = seq.to(self.device)   # (B, T, F)
                mask = mask.to(self.device)  # (B, T)

                recon = self.model(seq)
                loss  = self.criterion(recon, seq, mask)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grad_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip
                        )
                    self.optimizer.step()

                total_loss    += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)

    def _load_checkpoint(self, path: str, result: TrainingResult) -> int:
        """Load optimizer state from an extended checkpoint if available.

        Falls back to model-only checkpoint (produced by ``GRUAutoencoder.save``).

        Returns
        -------
        int
            The epoch to resume from (0-indexed).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])

        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "training_result" in ckpt:
            saved = ckpt["training_result"]
            result.train_losses  = saved.get("train_losses", [])
            result.val_losses    = saved.get("val_losses", [])
            result.epoch_times   = saved.get("epoch_times", [])
            result.best_val_loss = saved.get("best_val_loss", float("inf"))
            result.best_epoch    = saved.get("best_epoch", 0)

        return ckpt.get("epoch", 0)

    def save_training_history(
        self,
        result: TrainingResult,
        output_dir: str = "outputs",
    ) -> None:
        """Persist the training history to ``training_history.json``.

        Parameters
        ----------
        result : TrainingResult
        output_dir : str
        """
        ensure_dir(output_dir)
        path = os.path.join(output_dir, "training_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.as_dict(), f, indent=2)
        logger.info("Training history saved → %s", path)
