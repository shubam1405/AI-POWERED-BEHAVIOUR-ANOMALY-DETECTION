"""
inference.py – Real-time anomaly scoring for new sessions.

Given a single session (as a raw sequence array or a
:class:`~anomaly_detection.dataset_loader.SessionRecord`), the
:class:`InferenceEngine` runs the trained GRU Autoencoder, computes the
reconstruction error, compares it against the stored threshold, and returns
a structured result dict suitable for downstream consumers (SHAP, Copilot).

Improvements
------------
* Automatically loads the optimal threshold from the checkpoint (saved by
  :meth:`~anomaly_detection.trainer.GRUTrainer.save_checkpoint_with_meta`).
* Automatically loads and applies the StandardScaler from the checkpoint
  so sequences are normalised before inference — consistent with training.
* Falls back to a configurable default threshold if the checkpoint does
  not embed one (backwards-compatible with old checkpoints).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from anomaly_detection.gru_autoencoder import GRUAutoencoder
from anomaly_detection.utils import get_device

logger = logging.getLogger("AnomalyDetection.Inference")

# Type alias for a single raw sequence (list-of-lists or ndarray)
RawSequence = Union[List[List[float]], np.ndarray]

# Default threshold used when the checkpoint does not embed one
_DEFAULT_THRESHOLD = 0.05


class InferenceEngine:
    """Scores new sessions using a trained GRU Autoencoder.

    Parameters
    ----------
    model_path : str
        Path to the saved model checkpoint (``models/gru_autoencoder.pt``).
    threshold : float, optional
        Override the threshold.  If *None* (default), the threshold embedded
        in the checkpoint is used.  If neither is available, falls back to
        ``_DEFAULT_THRESHOLD``.
    device : torch.device, optional
        Defaults to automatic CPU/GPU selection.

    Example
    -------
    >>> engine = InferenceEngine("models/gru_autoencoder.pt")
    >>> result = engine.score_sequence(session_id="S-001", sequence=seq_array)
    >>> print(result["prediction"])   # "Normal" or "Anomalous"
    """

    def __init__(
        self,
        model_path: str = "models/gru_autoencoder.pt",
        threshold: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device       = device or get_device()
        self._err_min: float = 0.0
        self._err_max: float = 1.0
        self._scaler      = None   # StandardScaler or None

        # Load checkpoint to extract threshold and scaler BEFORE building model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Resolve threshold: explicit arg → checkpoint → default
        if threshold is not None:
            self.threshold = threshold
            logger.info("Using provided threshold=%.6f", self.threshold)
        elif "threshold" in checkpoint:
            self.threshold = float(checkpoint["threshold"])
            method = checkpoint.get("threshold_method", "unknown")
            logger.info(
                "Loaded threshold=%.6f from checkpoint (method=%s)",
                self.threshold, method,
            )
        else:
            self.threshold = _DEFAULT_THRESHOLD
            logger.warning(
                "Checkpoint has no embedded threshold. "
                "Using default threshold=%.6f. "
                "Run train_gru.py to embed the optimal threshold.",
                self.threshold,
            )

        # Load scaler if embedded in checkpoint
        if "scaler" in checkpoint:
            try:
                self._scaler = pickle.loads(checkpoint["scaler"])
                logger.info(
                    "StandardScaler loaded from checkpoint "
                    "(mean shape=%s).",
                    self._scaler.mean_.shape,
                )
            except Exception as exc:
                logger.warning("Failed to deserialise scaler from checkpoint: %s", exc)
                self._scaler = None
        else:
            logger.info(
                "No scaler found in checkpoint. "
                "Sequences will be used without normalisation."
            )

        # Build the model
        logger.info("Loading inference model from %s …", model_path)
        self.model = GRUAutoencoder.load(model_path, device=self.device)
        self.model.eval()

    def calibrate(
        self,
        err_min: float,
        err_max: float,
    ) -> "InferenceEngine":
        """Store the training-set min/max errors for anomaly-score normalisation.

        Parameters
        ----------
        err_min : float
            Minimum reconstruction error seen on the training set.
        err_max : float
            Maximum reconstruction error seen on the training set.

        Returns
        -------
        self
        """
        self._err_min = err_min
        self._err_max = max(err_max, err_min + 1e-9)
        logger.debug("Calibration: err_min=%.6f  err_max=%.6f", err_min, err_max)
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_sequence(
        self,
        session_id: str,
        sequence: RawSequence,
        mask: Optional[RawSequence] = None,
    ) -> Dict[str, Any]:
        """Score a single session sequence.

        Parameters
        ----------
        session_id : str
            Identifier for the session (used in the output dict).
        sequence : array-like, shape ``(seq_len, feature_dim)``
            The event sequence tensor.
        mask : array-like, shape ``(seq_len,)``, optional
            Attention mask (1 = real event, 0 = padding).  If *None*, all
            time-steps are treated as real.

        Returns
        -------
        dict with keys:
            ``session_id``, ``reconstruction_error``, ``anomaly_score``,
            ``threshold``, ``prediction``, ``is_anomalous``
        """
        seq_arr = np.array(sequence, dtype=np.float32)
        if seq_arr.ndim != 2:
            raise ValueError(
                f"Sequence must be 2-D (seq_len × feature_dim), got shape {seq_arr.shape}."
            )

        # Apply scaler if available (consistent with training normalisation)
        if self._scaler is not None:
            seq_arr = self._scaler.transform(seq_arr).astype(np.float32)

        if mask is not None:
            msk_arr = np.array(mask, dtype=np.float32)
        else:
            msk_arr = np.ones(seq_arr.shape[0], dtype=np.float32)

        # Add batch dimension
        seq_t = torch.tensor(seq_arr).unsqueeze(0).to(self.device)   # (1, T, F)
        msk_t = torch.tensor(msk_arr).unsqueeze(0).to(self.device)   # (1, T)

        with torch.no_grad():
            recon = self.model(seq_t)

        # Per-sample masked MSE reconstruction error
        sq_err  = (recon - seq_t) ** 2
        msk_exp = msk_t.unsqueeze(-1).expand_as(sq_err)
        err_val = float((sq_err * msk_exp).sum() / msk_exp.sum().clamp(min=1.0))

        anomaly_score = self._normalise(err_val)
        is_anomalous  = int(err_val > self.threshold)

        result: Dict[str, Any] = {
            "session_id":           session_id,
            "reconstruction_error": round(err_val, 6),
            "anomaly_score":        round(anomaly_score, 6),
            "threshold":            round(self.threshold, 6),
            "prediction":           "Anomalous" if is_anomalous else "Normal",
            "is_anomalous":         is_anomalous,
        }

        logger.debug(
            "Session %s  error=%.6f  score=%.4f  prediction=%s",
            session_id, err_val, anomaly_score, result["prediction"],
        )
        return result

    def score_batch(
        self,
        sessions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Score multiple sessions.

        Parameters
        ----------
        sessions : list of dict, each with keys ``session_id``, ``sequence``
            and optionally ``mask``.

        Returns
        -------
        list of result dicts (same order as input).
        """
        results: List[Dict[str, Any]] = []
        for item in sessions:
            result = self.score_sequence(
                session_id=item["session_id"],
                sequence=item["sequence"],
                mask=item.get("mask"),
            )
            results.append(result)
        logger.info("Scored %d sessions.", len(results))
        return results

    def score_record(self, record: Any) -> Dict[str, Any]:
        """Score a :class:`~anomaly_detection.dataset_loader.SessionRecord`.

        Parameters
        ----------
        record : SessionRecord

        Returns
        -------
        dict
        """
        return self.score_sequence(
            session_id=record.session_id,
            sequence=record.sequence,
            mask=record.mask,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalise(self, error: float) -> float:
        """Map reconstruction error to ``[0, 1]`` anomaly score."""
        denom = self._err_max - self._err_min
        if denom < 1e-9:
            return 1.0 if error > self.threshold else 0.0
        return float(np.clip((error - self._err_min) / denom, 0.0, 1.0))
