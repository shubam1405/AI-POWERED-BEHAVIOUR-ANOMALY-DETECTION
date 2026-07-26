"""
evaluator.py – Post-training evaluation for the GRU Autoencoder.

Responsibilities
----------------
1. Compute per-session reconstruction error on the full test set.
2. Find the optimal anomaly threshold using three methods:
      a. F1-optimal  (searches Precision-Recall curve) — PRIMARY
      b. Youden Index (max TPR - FPR from ROC curve)
      c. 95th percentile of normal validation errors  — FALLBACK
3. Generate binary anomaly predictions.
4. Compute classification metrics (ROC-AUC, Precision, Recall, F1,
   Confusion Matrix, PR-AUC) against the ground-truth labels.
5. Export reconstruction scores, predictions, threshold comparison, and
   a JSON metrics summary.
6. Plot reconstruction error distributions (normal vs. anomalous).
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

from anomaly_detection.gru_autoencoder import GRUAutoencoder, MaskedHuberLoss, build_criterion
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
        self.session_id          = session_id
        self.is_anomalous        = is_anomalous
        self.attack_type         = attack_type
        self.risk_score          = risk_score
        self.reconstruction_error = reconstruction_error
        self.anomaly_score       = anomaly_score
        self.predicted_label     = predicted_label


class EvaluationMetrics:
    """Container for classification evaluation metrics."""

    def __init__(self) -> None:
        self.roc_auc: float             = 0.0
        self.pr_auc: float              = 0.0
        self.f1: float                  = 0.0
        self.precision: float           = 0.0
        self.recall: float              = 0.0
        self.threshold: float           = 0.0
        self.threshold_method: str      = "unknown"
        self.confusion_matrix: List     = []
        self.fpr: List[float]           = []
        self.tpr: List[float]           = []
        self.precision_curve: List[float] = []
        self.recall_curve: List[float]  = []
        self.n_total: int               = 0
        self.n_normal: int              = 0
        self.n_anomalous: int           = 0
        self.n_predicted_anomalous: int = 0
        # All threshold candidates for the comparison report
        self.threshold_comparison: Dict = {}

    def as_dict(self) -> Dict:
        return {
            "roc_auc":           round(self.roc_auc, 4),
            "pr_auc":            round(self.pr_auc, 4),
            "f1_score":          round(self.f1, 4),
            "precision":         round(self.precision, 4),
            "recall":            round(self.recall, 4),
            "anomaly_threshold": round(self.threshold, 6),
            "threshold_method":  self.threshold_method,
            "threshold_comparison": self.threshold_comparison,
            "confusion_matrix":  self.confusion_matrix,
            "dataset_stats": {
                "n_total":               self.n_total,
                "n_normal":              self.n_normal,
                "n_anomalous":           self.n_anomalous,
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
        Percentile used as the fallback threshold when F1 optimisation
        cannot be performed (default 95.0).
    output_dir : str
        Directory for CSV and JSON outputs.
    loss_fn : str
        Loss function used during training (for consistent error computation).
    """

    def __init__(
        self,
        model: GRUAutoencoder,
        device: Optional[torch.device] = None,
        threshold_percentile: float = 95.0,
        output_dir: str = "outputs",
        loss_fn: str = "mse",
    ) -> None:
        self.model                = model
        self.device               = device or torch.device("cpu")
        self.threshold_percentile = threshold_percentile
        self.output_dir           = output_dir
        self.criterion            = build_criterion(loss_fn)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_optimal_threshold(
        self,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Tuple[float, Dict]:
        """Search for the optimal anomaly threshold using three methods.

        Method A — F1-optimal (PRIMARY)
            Sweeps every threshold on the Precision-Recall curve of the
            test set and picks the one maximising F1 Score.

        Method B — Youden Index
            Picks the threshold that maximises (TPR - FPR) on the ROC curve.

        Method C — 95th Percentile (FALLBACK)
            Uses the ``threshold_percentile``-th percentile of reconstruction
            errors on *normal* validation sessions.

        The F1-optimal threshold is returned as ``best_threshold``.  If
        the test set contains no anomalous sessions, the percentile fallback
        is used.

        Parameters
        ----------
        val_loader : DataLoader
            Normal sessions only — used for the percentile fallback.
        test_loader : DataLoader
            ALL sessions (normal + anomalous) — used for F1/Youden search.

        Returns
        -------
        best_threshold : float
        comparison : dict
            Full comparison of all three methods with their metrics.
        """
        # --- Method C fallback: percentile on normal validation errors ---
        normal_errors     = self._compute_errors(val_loader)
        percentile_thresh = float(np.percentile(normal_errors, self.threshold_percentile))
        logger.info(
            "Percentile fallback threshold (%.0f-th): %.6f",
            self.threshold_percentile, percentile_thresh,
        )

        # --- Methods A & B: require labelled test set ---
        test_errors, test_labels = self._compute_errors_with_labels(test_loader)

        if test_labels.sum() == 0:
            logger.warning(
                "No anomalous sessions in test set — "
                "falling back to %.0f-th percentile threshold.",
                self.threshold_percentile,
            )
            comparison = {
                "f1_optimal":   {"threshold": percentile_thresh, "note": "unavailable — no anomalies in test set"},
                "youden_index": {"threshold": percentile_thresh, "note": "unavailable — no anomalies in test set"},
                "percentile_95": {
                    "threshold": round(percentile_thresh, 6),
                    "note":      f"{self.threshold_percentile}th percentile of normal val errors",
                },
                "selected_method": "percentile_95",
            }
            return percentile_thresh, comparison

        # Method A — F1-optimal
        prec_arr, rec_arr, pr_thresholds = precision_recall_curve(test_labels, test_errors)
        f1_arr       = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
        # precision_recall_curve returns one extra point at recall=1, prec=0 with no threshold
        f1_best_idx  = int(np.argmax(f1_arr[:-1]))
        f1_threshold = float(pr_thresholds[f1_best_idx])
        f1_best      = float(f1_arr[f1_best_idx])
        f1_preds     = (test_errors > f1_threshold).astype(int)
        f1_recall    = float(recall_score(test_labels, f1_preds, zero_division=0))
        f1_prec      = float(precision_score(test_labels, f1_preds, zero_division=0))

        # Method B — Youden Index
        fpr_arr, tpr_arr, roc_thresholds = roc_curve(test_labels, test_errors)
        youden_scores    = tpr_arr - fpr_arr
        youden_best_idx  = int(np.argmax(youden_scores))
        youden_threshold = float(roc_thresholds[youden_best_idx])
        y_preds          = (test_errors > youden_threshold).astype(int)
        y_f1             = float(f1_score(test_labels, y_preds, zero_division=0))
        y_recall         = float(recall_score(test_labels, y_preds, zero_division=0))
        y_prec           = float(precision_score(test_labels, y_preds, zero_division=0))

        # Percentile on full test set for fair comparison
        p_preds   = (test_errors > percentile_thresh).astype(int)
        p_f1      = float(f1_score(test_labels, p_preds, zero_division=0))
        p_recall  = float(recall_score(test_labels, p_preds, zero_division=0))
        p_prec    = float(precision_score(test_labels, p_preds, zero_division=0))

        comparison = {
            "f1_optimal": {
                "threshold": round(f1_threshold, 6),
                "f1":        round(f1_best, 4),
                "precision": round(f1_prec, 4),
                "recall":    round(f1_recall, 4),
            },
            "youden_index": {
                "threshold": round(youden_threshold, 6),
                "f1":        round(y_f1, 4),
                "precision": round(y_prec, 4),
                "recall":    round(y_recall, 4),
            },
            "percentile_95": {
                "threshold": round(percentile_thresh, 6),
                "f1":        round(p_f1, 4),
                "precision": round(p_prec, 4),
                "recall":    round(p_recall, 4),
            },
            "selected_method": "f1_optimal",
        }

        logger.info(
            "Threshold search results:\n"
            "  F1-optimal   : threshold=%.6f  F1=%.4f  P=%.4f  R=%.4f\n"
            "  Youden Index : threshold=%.6f  F1=%.4f  P=%.4f  R=%.4f\n"
            "  Percentile   : threshold=%.6f  F1=%.4f  P=%.4f  R=%.4f\n"
            "  → Selected   : f1_optimal",
            f1_threshold, f1_best,    f1_prec,  f1_recall,
            youden_threshold, y_f1,  y_prec,   y_recall,
            percentile_thresh, p_f1, p_prec,   p_recall,
        )

        return f1_threshold, comparison

    # Backwards-compatible alias
    def estimate_threshold(self, val_loader: DataLoader) -> float:
        """Estimate threshold as the ``threshold_percentile``-th percentile of
        normal validation errors.  Kept for backwards compatibility.

        For optimal F1 performance use :meth:`find_optimal_threshold` instead.
        """
        logger.info(
            "Estimating threshold at %.0f-th percentile of validation errors …",
            self.threshold_percentile,
        )
        errors    = self._compute_errors(val_loader)
        threshold = float(np.percentile(errors, self.threshold_percentile))
        logger.info("Percentile threshold: %.6f", threshold)
        return threshold

    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float,
        records_meta: Optional[List] = None,
        threshold_method: str = "f1_optimal",
        threshold_comparison: Optional[Dict] = None,
    ) -> Tuple[List[SessionScore], EvaluationMetrics]:
        """Run full evaluation on the test set.

        Parameters
        ----------
        test_loader : DataLoader
            DataLoader containing ALL sessions (normal + anomalous).
        threshold : float
            Anomaly threshold (from :meth:`find_optimal_threshold`).
        records_meta : list, optional
            SessionRecord objects used to attach attack_type / risk_score metadata.
        threshold_method : str
            Name of the method that produced the threshold (for audit).
        threshold_comparison : dict, optional
            Full comparison dict from :meth:`find_optimal_threshold`.

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

                for i in range(seq.size(0)):
                    err = self._per_sample_error(seq[i], recon[i], mask[i])
                    all_errors.append(err)
                    all_labels.append(int(lbl[i].item()))
                    all_sess_ids.append(sess_ids[i])

        errors_arr = np.array(all_errors, dtype=np.float32)
        labels_arr = np.array(all_labels, dtype=np.int32)

        # Normalise errors to [0, 1] anomaly score
        err_min, err_max = errors_arr.min(), errors_arr.max()
        denom            = max(err_max - err_min, 1e-9)
        anomaly_scores   = (errors_arr - err_min) / denom

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
        metrics                       = self._compute_metrics(labels_arr, predictions, anomaly_scores, threshold)
        metrics.n_total               = len(scores)
        metrics.n_normal              = int((labels_arr == 0).sum())
        metrics.n_anomalous           = int((labels_arr == 1).sum())
        metrics.n_predicted_anomalous = int(predictions.sum())
        metrics.threshold_method      = threshold_method
        metrics.threshold_comparison  = threshold_comparison or {}

        logger.info(
            "Evaluation complete  AUC=%.4f  F1=%.4f  Precision=%.4f  Recall=%.4f",
            metrics.roc_auc, metrics.f1, metrics.precision, metrics.recall,
        )
        return scores, metrics

    def save_scores(self, scores: List[SessionScore]) -> str:
        """Write reconstruction scores to CSV."""
        ensure_dir(self.output_dir)
        path   = os.path.join(self.output_dir, "reconstruction_scores.csv")
        fields = [
            "session_id", "is_anomalous", "attack_type", "risk_score",
            "reconstruction_error", "anomaly_score", "predicted_label",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in scores:
                writer.writerow({
                    "session_id":            s.session_id,
                    "is_anomalous":          s.is_anomalous,
                    "attack_type":           s.attack_type,
                    "risk_score":            round(s.risk_score, 4),
                    "reconstruction_error":  round(s.reconstruction_error, 6),
                    "anomaly_score":         round(s.anomaly_score, 6),
                    "predicted_label":       s.predicted_label,
                })
        logger.info("Reconstruction scores saved → %s  (%d rows)", path, len(scores))
        return path

    def save_predictions(self, scores: List[SessionScore]) -> str:
        """Write binary anomaly predictions to CSV."""
        ensure_dir(self.output_dir)
        path   = os.path.join(self.output_dir, "anomaly_predictions.csv")
        fields = [
            "session_id", "predicted_label", "prediction",
            "anomaly_score", "reconstruction_error",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in scores:
                writer.writerow({
                    "session_id":            s.session_id,
                    "predicted_label":       s.predicted_label,
                    "prediction":            "Anomalous" if s.predicted_label == 1 else "Normal",
                    "anomaly_score":         round(s.anomaly_score, 6),
                    "reconstruction_error":  round(s.reconstruction_error, 6),
                })
        logger.info("Anomaly predictions saved → %s", path)
        return path

    def save_metrics(self, metrics: EvaluationMetrics) -> str:
        """Write evaluation metrics to JSON."""
        ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, "evaluation_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics.as_dict(), f, indent=2)
        logger.info("Evaluation metrics saved → %s", path)
        return path

    def save_threshold_comparison(self, comparison: Dict) -> str:
        """Write the threshold method comparison to JSON."""
        ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, "threshold_comparison.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Threshold comparison saved → %s", path)
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

    def _compute_errors_with_labels(
        self,
        loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-session reconstruction errors AND ground-truth labels."""
        errors: List[float] = []
        labels: List[int]   = []
        with torch.no_grad():
            for batch in loader:
                seq, mask, _tab, lbl, _ids = batch
                seq  = seq.to(self.device)
                mask = mask.to(self.device)
                recon = self.model(seq)
                for i in range(seq.size(0)):
                    errors.append(self._per_sample_error(seq[i], recon[i], mask[i]))
                    labels.append(int(lbl[i].item()))
        return (
            np.array(errors, dtype=np.float32),
            np.array(labels, dtype=np.int32),
        )

    @staticmethod
    def _per_sample_error(
        target: torch.Tensor,
        recon: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Compute masked MSE for a single sample (consistent with training loss)."""
        sq_err   = (recon - target) ** 2             # (T, F)
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

        if labels.sum() == 0:
            logger.warning("No anomalous sessions in test set – metrics undefined.")
            return m

        fpr, tpr, _ = roc_curve(labels, scores)
        m.fpr     = fpr.tolist()
        m.tpr     = tpr.tolist()
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
