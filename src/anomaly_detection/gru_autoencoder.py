"""
gru_autoencoder.py – GRU Encoder-Decoder Autoencoder for session anomaly detection.

Architecture
------------

    Input (batch, seq_len, feature_dim)
         │
         ▼
    ┌─────────────────────┐
    │   GRU Encoder       │   num_layers, hidden_size, dropout
    │                     │
    │   → last hidden h   │   shape: (batch, hidden_size)
    └──────────┬──────────┘
               │  latent representation
               ▼
    ┌─────────────────────┐
    │  Latent Projection  │   hidden_size → latent_dim (optional bottleneck)
    └──────────┬──────────┘
               │
               ▼  repeat across seq_len
    ┌─────────────────────┐
    │   GRU Decoder       │   num_layers, hidden_size, dropout
    │                     │
    │   → all hidden h    │   shape: (batch, seq_len, hidden_size)
    └──────────┬──────────┘
               │
               ▼
    ┌─────────────────────┐
    │  Output Projection  │   hidden_size → feature_dim
    └──────────┬──────────┘
               │
               ▼
    Reconstructed Sequence (batch, seq_len, feature_dim)

Loss
----
Masked MSE – only real (non-padded) time-steps contribute to the loss.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger("AnomalyDetection.GRUAutoencoder")


# ---------------------------------------------------------------------------
# Weight initialisation helper
# ---------------------------------------------------------------------------

def _init_weights(module: nn.Module) -> None:
    """Xavier-uniform for linear layers; orthogonal for GRU weights."""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class GRUEncoder(nn.Module):
    """Multi-layer GRU Encoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of each time-step feature vector (``feature_dim``).
    hidden_size : int
        Number of GRU hidden units per layer.
    num_layers : int
        Number of stacked GRU layers.
    dropout : float
        Dropout probability applied between GRU layers (0 = disabled).
    bidirectional : bool
        If *True*, use a bidirectional GRU; the hidden state is merged
        by concatenation and projected back to ``hidden_size``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # Merge bidirectional hidden states → hidden_size
        if bidirectional:
            self.merge = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        else:
            self.merge = nn.Identity()

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a padded sequence batch.

        Parameters
        ----------
        x : Tensor, shape ``(batch, seq_len, input_dim)``

        Returns
        -------
        latent : Tensor, shape ``(batch, hidden_size)``
            Last-layer, last-timestep hidden state (merged if bidirectional).
        """
        _, hidden = self.gru(x)  # hidden: (num_layers*dirs, batch, hidden)
        # Take only the last layer's hidden state
        if self.bidirectional:
            fwd = hidden[-2]   # forward  last layer
            bwd = hidden[-1]   # backward last layer
            latent = self.merge(torch.cat([fwd, bwd], dim=-1))
        else:
            latent = hidden[-1]  # (batch, hidden_size)

        return self.layer_norm(latent)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class GRUDecoder(nn.Module):
    """Multi-layer GRU Decoder.

    The latent vector is repeated ``seq_len`` times to form the decoder input,
    allowing the decoder to attend to the full latent representation at each
    time-step (seq2seq without attention).

    Parameters
    ----------
    hidden_size : int
    output_dim : int
        Dimensionality of the reconstructed feature vector.
    num_layers : int
    dropout : float
    seq_len : int
        The fixed output sequence length (must match encoder input length).
    """

    def __init__(
        self,
        hidden_size: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 50,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, output_dim)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation back to a sequence.

        Parameters
        ----------
        latent : Tensor, shape ``(batch, hidden_size)``

        Returns
        -------
        recon : Tensor, shape ``(batch, seq_len, output_dim)``
        """
        # Repeat latent across time-steps: (batch, seq_len, hidden_size)
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        output, _ = self.gru(decoder_input)   # (batch, seq_len, hidden_size)
        recon = self.output_proj(output)       # (batch, seq_len, output_dim)
        return recon


# ---------------------------------------------------------------------------
# Full autoencoder
# ---------------------------------------------------------------------------

class GRUAutoencoder(nn.Module):
    """GRU Encoder-Decoder Autoencoder for session anomaly detection.

    Parameters
    ----------
    input_dim : int
        Feature dimension of each event time-step (default 21).
    hidden_size : int
        Number of GRU hidden units (default 128).
    num_layers : int
        Number of stacked GRU layers in both encoder and decoder (default 2).
    dropout : float
        Dropout probability between GRU layers (default 0.3).
    seq_len : int
        Fixed sequence length the model expects (default 50).
    bidirectional_encoder : bool
        Whether to use a bidirectional GRU in the encoder (default False).

    Example
    -------
    >>> model = GRUAutoencoder(input_dim=21, hidden_size=128, seq_len=50)
    >>> x = torch.randn(32, 50, 21)
    >>> recon = model(x)
    >>> recon.shape
    torch.Size([32, 50, 21])
    """

    def __init__(
        self,
        input_dim: int = 21,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        seq_len: int = 50,
        bidirectional_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Optional input projection for dimensionality expansion
        self.input_proj = nn.Linear(input_dim, hidden_size) if input_dim != hidden_size else nn.Identity()

        self.encoder = GRUEncoder(
            input_dim=hidden_size if input_dim != hidden_size else input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional_encoder,
        )

        self.decoder = GRUDecoder(
            hidden_size=hidden_size,
            output_dim=input_dim,   # reconstruct original feature space
            num_layers=num_layers,
            dropout=dropout,
            seq_len=seq_len,
        )

        # Initialise weights
        self.apply(_init_weights)
        logger.info(
            "GRUAutoencoder initialised  input_dim=%d  hidden=%d  layers=%d  seq_len=%d  params=%s",
            input_dim, hidden_size, num_layers, seq_len,
            f"{sum(p.numel() for p in self.parameters()):,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode.

        Parameters
        ----------
        x : Tensor, shape ``(batch, seq_len, input_dim)``

        Returns
        -------
        recon : Tensor, shape ``(batch, seq_len, input_dim)``
        """
        projected = self.input_proj(x) if not isinstance(self.input_proj, nn.Identity) else x
        latent = self.encoder(projected)
        recon = self.decoder(latent)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation only.

        Parameters
        ----------
        x : Tensor, shape ``(batch, seq_len, input_dim)``

        Returns
        -------
        latent : Tensor, shape ``(batch, hidden_size)``
        """
        projected = self.input_proj(x) if not isinstance(self.input_proj, nn.Identity) else x
        return self.encoder(projected)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the model state dictionary and hyper-parameters.

        Parameters
        ----------
        path : str
            File path for the saved checkpoint (e.g. ``"models/gru_autoencoder.pt"``).
        """
        from anomaly_detection.utils import ensure_dir
        ensure_dir(Path(path).parent)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "hparams": {
                    "input_dim": self.input_dim,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "seq_len": self.seq_len,
                },
            },
            path,
        )
        logger.info("Model checkpoint saved → %s", path)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "GRUAutoencoder":
        """Load a model from a checkpoint saved by :meth:`save`.

        Parameters
        ----------
        path : str
        device : torch.device, optional
            Target device.  Defaults to CPU.

        Returns
        -------
        GRUAutoencoder
        """
        if device is None:
            device = torch.device("cpu")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        hparams = checkpoint["hparams"]
        model = cls(**hparams)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        model.eval()
        logger.info("Model checkpoint loaded ← %s  (device=%s)", path, device)
        return model


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class MaskedMSELoss(nn.Module):
    """MSE loss that ignores padded time-steps.

    Parameters
    ----------
    reduction : str
        ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked reconstruction MSE.

        Parameters
        ----------
        recon  : Tensor, shape ``(batch, seq_len, feature_dim)``
        target : Tensor, shape ``(batch, seq_len, feature_dim)``
        mask   : Tensor, shape ``(batch, seq_len)``  – 1 = real, 0 = pad

        Returns
        -------
        loss : scalar Tensor
        """
        # Per-element squared error
        sq_err = (recon - target) ** 2                   # (batch, seq_len, feat)
        # Expand mask to feature dimension
        mask_expanded = mask.unsqueeze(-1).expand_as(sq_err)  # (batch, seq_len, feat)
        masked_sq_err = sq_err * mask_expanded

        if self.reduction == "mean":
            denom = mask_expanded.sum().clamp(min=1.0)
            return masked_sq_err.sum() / denom
        else:
            return masked_sq_err.sum()
