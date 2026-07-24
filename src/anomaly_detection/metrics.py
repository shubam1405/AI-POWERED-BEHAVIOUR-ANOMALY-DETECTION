"""
metrics.py – Visualization and plot generation for the anomaly detection engine.

Generates:
    - Training loss curve (train + validation)
    - Reconstruction error distribution (normal vs anomalous)
    - ROC curve
    - Precision-Recall curve
    - Confusion matrix heatmap

All plots are saved to ``outputs/plots/`` as high-resolution PNG files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from anomaly_detection.utils import ensure_dir

logger = logging.getLogger("AnomalyDetection.Metrics")

# ---------------------------------------------------------------------------
# Plot style configuration
# ---------------------------------------------------------------------------

_STYLE = {
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "axes.grid":         True,
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "text.color":        "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
}

_NORMAL_COLOR    = "#58a6ff"
_ANOMALOUS_COLOR = "#f78166"
_ACCENT_COLOR    = "#3fb950"
_WARN_COLOR      = "#d29922"

DPI = 150


# ---------------------------------------------------------------------------
# Main plotter class
# ---------------------------------------------------------------------------

class AnomalyMetricsPlotter:
    """Generates and saves all diagnostic plots.

    Parameters
    ----------
    output_dir : str
        Directory where plots will be saved (default ``"outputs/plots"``).
    """

    def __init__(self, output_dir: str = "outputs/plots") -> None:
        self.output_dir = ensure_dir(output_dir)

    # ------------------------------------------------------------------
    # 1. Training loss curves
    # ------------------------------------------------------------------

    def plot_loss_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        best_epoch: Optional[int] = None,
    ) -> str:
        """Plot training and validation loss over epochs.

        Parameters
        ----------
        train_losses : list of float
        val_losses   : list of float
        best_epoch   : int, optional
            If provided, marks the best epoch with a vertical dashed line.

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "loss_curve.png")
        epochs = list(range(1, len(train_losses) + 1))

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(epochs, train_losses, color=_NORMAL_COLOR,    linewidth=2, label="Train Loss")
            ax.plot(epochs, val_losses,   color=_ANOMALOUS_COLOR, linewidth=2, label="Validation Loss")

            if best_epoch is not None:
                ax.axvline(best_epoch, color=_ACCENT_COLOR, linestyle="--",
                           linewidth=1.5, label=f"Best Epoch ({best_epoch})", alpha=0.8)

            ax.set_title("GRU Autoencoder — Training & Validation Loss")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Masked MSE Loss")
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Loss curve saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 2. Reconstruction error distribution
    # ------------------------------------------------------------------

    def plot_reconstruction_error_distribution(
        self,
        normal_errors: List[float],
        anomalous_errors: List[float],
        threshold: float,
        bins: int = 60,
    ) -> str:
        """Histogram of reconstruction errors: normal vs anomalous.

        Parameters
        ----------
        normal_errors    : list of float
        anomalous_errors : list of float
        threshold        : float – the decision boundary
        bins             : int

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "reconstruction_error_distribution.png")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(11, 5))

            all_errors = normal_errors + anomalous_errors
            e_min = min(all_errors) if all_errors else 0.0
            e_max = max(all_errors) if all_errors else 1.0
            bin_edges = np.linspace(e_min, e_max, bins + 1)

            ax.hist(normal_errors,    bins=bin_edges, color=_NORMAL_COLOR,
                    alpha=0.7, label=f"Normal (n={len(normal_errors):,})")
            ax.hist(anomalous_errors, bins=bin_edges, color=_ANOMALOUS_COLOR,
                    alpha=0.7, label=f"Anomalous (n={len(anomalous_errors):,})")

            ax.axvline(threshold, color=_WARN_COLOR, linestyle="--", linewidth=2,
                       label=f"Threshold ({threshold:.4f})")

            ax.set_title("Reconstruction Error Distribution — Normal vs Anomalous")
            ax.set_xlabel("Reconstruction Error (Masked MSE)")
            ax.set_ylabel("Session Count")
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Reconstruction error distribution saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 3. ROC curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        fpr: List[float],
        tpr: List[float],
        roc_auc: float,
    ) -> str:
        """Plot the ROC curve.

        Parameters
        ----------
        fpr     : list of float
        tpr     : list of float
        roc_auc : float

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "roc_curve.png")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7, 6))

            ax.plot(fpr, tpr, color=_ACCENT_COLOR, linewidth=2.5,
                    label=f"GRU-AE (AUC = {roc_auc:.4f})")
            ax.plot([0, 1], [0, 1], color="#8b949e", linestyle="--",
                    linewidth=1, label="Random classifier")

            ax.fill_between(fpr, tpr, alpha=0.08, color=_ACCENT_COLOR)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve — GRU Autoencoder")
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("ROC curve saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 4. Precision-Recall curve
    # ------------------------------------------------------------------

    def plot_precision_recall_curve(
        self,
        precision_vals: List[float],
        recall_vals: List[float],
        pr_auc: float,
    ) -> str:
        """Plot the Precision-Recall curve.

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "precision_recall_curve.png")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(7, 6))

            ax.plot(recall_vals, precision_vals, color=_ANOMALOUS_COLOR,
                    linewidth=2.5, label=f"GRU-AE (PR-AUC = {pr_auc:.4f})")
            ax.fill_between(recall_vals, precision_vals, alpha=0.08, color=_ANOMALOUS_COLOR)

            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve — GRU Autoencoder")
            ax.legend(loc="upper right")
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Precision-Recall curve saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 5. Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        cm: List[List[int]],
        labels: Optional[List[str]] = None,
    ) -> str:
        """Plot a heatmap of the confusion matrix.

        Parameters
        ----------
        cm     : list of list of int  [[TN, FP], [FN, TP]]
        labels : list of str, optional

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "confusion_matrix.png")
        cm_arr = np.array(cm)
        labels = labels or ["Normal", "Anomalous"]

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(6, 5))

            # Normalised colour map
            cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)
            cax = ax.imshow(cm_norm, interpolation="nearest",
                            cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

            tick_marks = np.arange(len(labels))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)

            thresh = cm_norm.max() / 2.0
            for i in range(cm_arr.shape[0]):
                for j in range(cm_arr.shape[1]):
                    color = "white" if cm_norm[i, j] > thresh else "#c9d1d9"
                    ax.text(j, i, f"{cm_arr[i, j]:,}",
                            ha="center", va="center", color=color, fontsize=12)

            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            ax.set_title("Confusion Matrix")
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Confusion matrix saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 6. Anomaly score distribution
    # ------------------------------------------------------------------

    def plot_anomaly_score_distribution(
        self,
        normal_scores: List[float],
        anomalous_scores: List[float],
    ) -> str:
        """KDE-style histogram of normalised anomaly scores.

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "anomaly_score_distribution.png")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(11, 5))
            bins = np.linspace(0, 1, 50)

            ax.hist(normal_scores,    bins=bins, color=_NORMAL_COLOR,
                    alpha=0.75, density=True, label=f"Normal (n={len(normal_scores):,})")
            ax.hist(anomalous_scores, bins=bins, color=_ANOMALOUS_COLOR,
                    alpha=0.75, density=True, label=f"Anomalous (n={len(anomalous_scores):,})")

            ax.set_title("Anomaly Score Distribution (Normalised Reconstruction Error)")
            ax.set_xlabel("Anomaly Score [0 – 1]")
            ax.set_ylabel("Density")
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Anomaly score distribution saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # Convenience: plot everything at once
    # ------------------------------------------------------------------

    def plot_all(
        self,
        train_losses: List[float],
        val_losses: List[float],
        best_epoch: int,
        scores_by_label: Dict,
        metrics: "EvaluationMetrics",  # type: ignore[name-defined]
    ) -> List[str]:
        """Generate all plots and return a list of saved file paths.

        Parameters
        ----------
        train_losses : list
        val_losses   : list
        best_epoch   : int
        scores_by_label : dict with keys ``"normal"`` and ``"anomalous"``,
            each containing a dict with ``"reconstruction_error"`` and
            ``"anomaly_score"`` lists.
        metrics : EvaluationMetrics

        Returns
        -------
        list of str – paths to saved PNG files.
        """
        saved_paths: List[str] = []

        saved_paths.append(
            self.plot_loss_curves(train_losses, val_losses, best_epoch)
        )
        saved_paths.append(
            self.plot_reconstruction_error_distribution(
                normal_errors=scores_by_label["normal"]["reconstruction_error"],
                anomalous_errors=scores_by_label["anomalous"]["reconstruction_error"],
                threshold=metrics.threshold,
            )
        )
        saved_paths.append(
            self.plot_roc_curve(metrics.fpr, metrics.tpr, metrics.roc_auc)
        )
        saved_paths.append(
            self.plot_precision_recall_curve(
                metrics.precision_curve,
                metrics.recall_curve,
                metrics.pr_auc,
            )
        )
        saved_paths.append(
            self.plot_confusion_matrix(metrics.confusion_matrix)
        )
        saved_paths.append(
            self.plot_anomaly_score_distribution(
                normal_scores=scores_by_label["normal"]["anomaly_score"],
                anomalous_scores=scores_by_label["anomalous"]["anomaly_score"],
            )
        )

        logger.info("All %d plots generated.", len(saved_paths))
        return saved_paths
