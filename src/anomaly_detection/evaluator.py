"""
evaluator.py – Post-training evaluation for the GRU Autoencoder.

Responsibilities
----------------
1. Compute per-session reconstruction error on the full test set.
2. Automatically estimate an anomaly threshold from the validation set.
3. Generate binary anomaly predictions.
4. Compute classification metrics (ROC-AUC, precision, recall, F1,
   confusion matrix) against the ground-truth ``is_anomalous`` labels.
5. Export reconstruction scores, predictions, and a JSON metrics summary.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from anomaly_detection.gru_autoencoder import GRUAutoencoder, MaskedMSELoss
from anomaly_detection.utils import ensure_dir

logger = logging.getLogger("AnomalyDetection.Evaluator")


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class SessionScore:
    """Reconstruction error and prediction for a single session."""

    __slots__ = (
        "session_id", "is_anomalous", "attack_type",
        "risk_score", "reconstruction_error", "anomaly_score",
        "predicted_label",
    )

    def __init__(
        self,
        session_id: str,
        is_anomalous: int,
        attack_type: str,
        risk_score: float,
        reconstruction_error: float,
        anomaly_score: float,
        predicted_label: int,
    ) -> None:
        self.session_id = session_id
        self.is_anomalous = is_anomalous
        self.attack_type = attack_type
        self.risk_score = risk_score
        self.reconstruction_error = reconstruction_error
        self.anomaly_score = anomaly_score
        self.predicted_label = predicted_label


class EvaluationMetrics:
    """Container for classification evaluation metrics."""

    def __init__(self) -> None:
        self.roc_auc: float = 0.0
        self.pr_auc: float = 0.0
        self.f1: float = 0.0
        self.precision: float = 0.0
        self.recall: float = 0.0
        self.threshold: float = 0.0
        self.confusion_matrix: List[List[int]] = []
        self.fpr: List[float] = []
        self.tpr: List[float] = []
        self.precision_curve: List[float] = []
        self.recall_curve: List[float] = []
        self.n_total: int = 0
        self.n_normal: int = 0
        self.n_anomalous: int = 0
        self.n_predicted_anomalous: int = 0

    def as_dict(self) -> Dict:
        return {
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "f1_score": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "anomaly_threshold": round(self.threshold, 6),
            "confusion_matrix": self.confusion_matrix,
            "dataset_stats": {
                "n_total": self.n_total,
                "n_normal": self.n_normal,
                "n_anomalous": self.n_anomalous,
                "n_predicted_anomalous": self.n_predicted_anomalous,
            },
        }


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class AnomalyEvaluator:
    """Evaluates a trained GRU Autoencoder on the full (normal + anomalous) test set.

    Parameters
    ----------
    model : GRUAutoencoder
        The trained model (loaded from best checkpoint).
    device : torch.device, optional
        Defaults to CPU.
    threshold_percentile : float
        Percentile of normal-session reconstruction errors on the validation
        set used to set the anomaly threshold (default 95.0).
    output_dir : str
        Directory for CSV and JSON outputs.
    """

    def __init__(
        self,
        model: GRUAutoencoder,
        device: Optional[torch.device] = None,
        threshold_percentile: float = 95.0,
        output_dir: str = "outputs",
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.threshold_percentile = threshold_percentile
        self.output_dir = output_dir
        self.criterion = MaskedMSELoss(reduction="mean")

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_threshold(self, val_loader: DataLoader) -> float:
        """Estimate the anomaly threshold from the validation set.

        Uses the ``threshold_percentile``-th percentile of reconstruction
        errors on normal (validation) sessions.  Since the validation set
        contains only normal sessions, this bounds the expected range of
        normal reconstruction error.

        Parameters
        ----------
        val_loader : DataLoader
            DataLoader containing only normal sessions.

        Returns
        -------
        float
            The estimated threshold.
        """
        logger.info(
            "Estimating threshold at %.0f-th percentile of validation errors …",
            self.threshold_percentile,
        )
        errors = self._compute_errors(val_loader)
        threshold = float(np.percentile(errors, self.threshold_percentile))
        logger.info("Estimated anomaly threshold: %.6f", threshold)
        return threshold

    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float,
        records_meta: Optional[List] = None,
    ) -> Tuple[List[SessionScore], EvaluationMetrics]:
        """Run full evaluation on the test set.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader containing ALL sessions (normal + anomalous).
        threshold : float
            Anomaly threshold (from :meth:`estimate_threshold`).
        records_meta : list, optional
            If provided, a list of :class:`~anomaly_detection.dataset_loader.SessionRecord`
            objects used to attach attack_type / risk_score metadata.

        Returns
        -------
        scores : list of SessionScore
        metrics : EvaluationMetrics
        """
        logger.info("Running evaluation on test set (threshold=%.6f) …", threshold)

        # Build a fast lookup for metadata by session_id
        meta_lookup: Dict[str, Dict] = {}
        if records_meta:
            for rec in records_meta:
                meta_lookup[rec.session_id] = {
                    "attack_type": rec.attack_type,
                    "risk_score":  rec.risk_score,
                }

        all_errors:   List[float] = []
        all_labels:   List[int]   = []
        all_sess_ids: List[str]   = []

        with torch.no_grad():
            for batch in test_loader:
                seq, mask, _tab, lbl, sess_ids = batch
                seq  = seq.to(self.device)
                mask = mask.to(self.device)

                recon = self.model(seq)

                # Per-session masked MSE
                for i in range(seq.size(0)):
                    err = self._per_sample_error(seq[i], recon[i], mask[i])
                    all_errors.append(err)
                    all_labels.append(int(lbl[i].item()))
                    all_sess_ids.append(sess_ids[i])

        errors_arr = np.array(all_errors, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int32)

        # Normalise errors to [0, 1] anomaly score
        err_min, err_max = errors_arr.min(), errors_arr.max()
        denom = max(err_max - err_min, 1e-9)
        anomaly_scores = (errors_arr - err_min) / denom

        predictions = (errors_arr > threshold).astype(np.int32)

        # Build SessionScore objects
        scores: List[SessionScore] = []
        for i, sid in enumerate(all_sess_ids):
            meta = meta_lookup.get(sid, {})
            scores.append(
                SessionScore(
                    session_id=sid,
                    is_anomalous=int(all_labels[i]),
                    attack_type=meta.get("attack_type", "None"),
                    risk_score=meta.get("risk_score", 0.0),
                    reconstruction_error=float(errors_arr[i]),
                    anomaly_score=float(anomaly_scores[i]),
                    predicted_label=int(predictions[i]),
                )
            )

        # Compute metrics
        metrics = self._compute_metrics(labels_arr, predictions, anomaly_scores, threshold)
        metrics.n_total             = len(scores)
        metrics.n_normal            = int((labels_arr == 0).sum())
        metrics.n_anomalous         = int((labels_arr == 1).sum())
        metrics.n_predicted_anomalous = int(predictions.sum())

        logger.info(
            "Evaluation complete  AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
            metrics.roc_auc, metrics.f1, metrics.precision, metrics.recall,
        )
        return scores, metrics

    def save_scores(self, scores: List[SessionScore]) -> str:
        """Write reconstruction scores to CSV.

        Parameters
        ----------
        scores : list of SessionScore

        Returns
        -------
        str : Path to saved file.
        """
        ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, "reconstruction_scores.csv")
        fields = [
            "session_id", "is_anomalous", "attack_type", "risk_score",
            "reconstruction_error", "anomaly_score", "predicted_label",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in scores:
                writer.writerow({
                    "session_id":           s.session_id,
                    "is_anomalous":         s.is_anomalous,
                    "attack_type":          s.attack_type,
                    "risk_score":           round(s.risk_score, 4),
                    "reconstruction_error": round(s.reconstruction_error, 6),
                    "anomaly_score":        round(s.anomaly_score, 6),
                    "predicted_label":      s.predicted_label,
                })
        logger.info("Reconstruction scores saved → %s  (%d rows)", path, len(scores))
        return path

    def save_predictions(self, scores: List[SessionScore]) -> str:
        """Write binary anomaly predictions to CSV.

        Returns
        -------
        str : Path to saved file.
        """
        ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, "anomaly_predictions.csv")
        fields = [
            "session_id", "predicted_label", "prediction",
            "anomaly_score", "reconstruction_error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in scores:
                writer.writerow({
                    "session_id":           s.session_id,
                    "predicted_label":      s.predicted_label,
                    "prediction":           "Anomalous" if s.predicted_label == 1 else "Normal",
                    "anomaly_score":        round(s.anomaly_score, 6),
                    "reconstruction_error": round(s.reconstruction_error, 6),
                })
        logger.info("Anomaly predictions saved → %s", path)
        return path

    def save_metrics(self, metrics: EvaluationMetrics) -> str:
        """Write evaluation metrics to JSON.

        Returns
        -------
        str : Path to saved file.
        """
        ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, "evaluation_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics.as_dict(), f, indent=2)
        logger.info("Evaluation metrics saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_errors(self, loader: DataLoader) -> np.ndarray:
        """Compute per-session reconstruction errors for all batches."""
        errors: List[float] = []
        with torch.no_grad():
            for batch in loader:
                seq, mask, _tab, _lbl, _ids = batch
                seq  = seq.to(self.device)
                mask = mask.to(self.device)
                recon = self.model(seq)
                for i in range(seq.size(0)):
                    errors.append(self._per_sample_error(seq[i], recon[i], mask[i]))
        return np.array(errors, dtype=np.float32)

    @staticmethod
    def _per_sample_error(
        target: torch.Tensor,
        recon: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Compute masked MSE for a single sample.

        Parameters
        ----------
        target : Tensor ``(seq_len, feature_dim)``
        recon  : Tensor ``(seq_len, feature_dim)``
        mask   : Tensor ``(seq_len,)``

        Returns
        -------
        float
        """
        sq_err = (recon - target) ** 2              # (T, F)
        mask_exp = mask.unsqueeze(-1).expand_as(sq_err)
        masked   = (sq_err * mask_exp).sum()
        denom    = mask_exp.sum().clamp(min=1.0)
        return float(masked / denom)

    @staticmethod
    def _compute_metrics(
        labels: np.ndarray,
        predictions: np.ndarray,
        scores: np.ndarray,
        threshold: float,
    ) -> EvaluationMetrics:
        m = EvaluationMetrics()
        m.threshold = threshold

        # Guard: need at least one positive class
        if labels.sum() == 0:
            logger.warning("No anomalous sessions in test set – metrics undefined.")
            return m

        fpr, tpr, _ = roc_curve(labels, scores)
        m.fpr = fpr.tolist()
        m.tpr = tpr.tolist()
        m.roc_auc = float(roc_auc_score(labels, scores))

        prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
        m.precision_curve = prec_curve.tolist()
        m.recall_curve    = rec_curve.tolist()
        m.pr_auc = float(auc(rec_curve, prec_curve))

        m.f1        = float(f1_score(labels, predictions, zero_division=0))
        m.precision = float(precision_score(labels, predictions, zero_division=0))
        m.recall    = float(recall_score(labels, predictions, zero_division=0))

        cm = confusion_matrix(labels, predictions)
        m.confusion_matrix = cm.tolist()

        return m
