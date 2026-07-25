"""
visualization.py – Dark-themed plots for the Attack Classification Engine.

Generates:
    1. Feature Importance (horizontal bar chart, top 20)
    2. Confusion Matrix (heatmap with normalised colour scale)
    3. Per-class ROC Curves
    4. Per-class Precision–Recall Curves
    5. Per-class Performance Chart (grouped bar: precision, recall, F1)

All plots use a dark GitHub-style theme and are saved at 150 DPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from attack_classification.utils import ensure_dir

logger = logging.getLogger("AttackClassification.Visualization")

# ---------------------------------------------------------------------------
# Style constants
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
    "font.size":         10,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
}

# Colour palette for up to 16 classes
_PALETTE = [
    "#58a6ff", "#f78166", "#3fb950", "#d29922",
    "#bc8cff", "#f0883e", "#a5d6ff", "#7ee787",
    "#ffa657", "#ff7b72", "#79c0ff", "#d2a8ff",
    "#56d364", "#e3b341", "#ff9bce", "#89929b",
]

DPI = 150


# ---------------------------------------------------------------------------
# Main plotter
# ---------------------------------------------------------------------------

class AttackVisualization:
    """Generates and saves diagnostic plots for attack classification.

    Parameters
    ----------
    output_dir : str
        Directory for saving plots.
    """

    def __init__(self, output_dir: str = "outputs/plots") -> None:
        self.output_dir = ensure_dir(output_dir)

    # ------------------------------------------------------------------
    # 1. Feature Importance
    # ------------------------------------------------------------------

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20,
    ) -> str:
        """Horizontal bar chart of top-N features by gain.

        Parameters
        ----------
        importance_df : pd.DataFrame
            Must have ``feature`` and ``gain`` columns.
        top_n : int

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "feature_importance.png")
        top = importance_df.head(top_n).sort_values("gain", ascending=True)

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 8))
            colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(top))]
            ax.barh(top["feature"], top["gain"], color=colours, height=0.7)
            ax.set_xlabel("Gain Importance")
            ax.set_title(f"Top {top_n} Feature Importance — XGBoost Attack Classifier")
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Feature importance plot saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 2. Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        cm: List[List[int]],
        class_names: List[str],
    ) -> str:
        """Heatmap of the confusion matrix.

        Parameters
        ----------
        cm : list of list of int
        class_names : list of str

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "confusion_matrix.png")
        cm_arr = np.array(cm)

        # Row-normalise for colour mapping
        row_sums = cm_arr.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1.0
        cm_norm = cm_arr / row_sums

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(14, 11))

            cax = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
            fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

            tick_marks = np.arange(len(class_names))
            ax.set_xticks(tick_marks)
            ax.set_yticks(tick_marks)
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(class_names, fontsize=8)

            thresh = cm_norm.max() / 2.0
            for i in range(cm_arr.shape[0]):
                for j in range(cm_arr.shape[1]):
                    colour = "white" if cm_norm[i, j] > thresh else "#c9d1d9"
                    val_str = f"{cm_arr[i, j]:,}" if cm_arr[i, j] > 0 else ""
                    ax.text(j, i, val_str,
                            ha="center", va="center", color=colour, fontsize=7)

            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            ax.set_title("Confusion Matrix — XGBoost Attack Classifier")
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Confusion matrix plot saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 3. ROC Curves (per-class)
    # ------------------------------------------------------------------

    def plot_roc_curves(
        self,
        per_class_roc: Dict[str, Dict[str, List[float]]],
        roc_auc: float,
    ) -> str:
        """Per-class ROC curves.

        Parameters
        ----------
        per_class_roc : dict  {class_name: {"fpr": [...], "tpr": [...]}}
        roc_auc : float  overall macro AUC

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "roc_curves.png")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 8))

            for i, (cls_name, curves) in enumerate(per_class_roc.items()):
                colour = _PALETTE[i % len(_PALETTE)]
                ax.plot(curves["fpr"], curves["tpr"],
                        color=colour, linewidth=1.5, alpha=0.8,
                        label=cls_name)

            ax.plot([0, 1], [0, 1], color="#8b949e", linestyle="--", linewidth=1)
            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curves (One-vs-Rest) — Macro AUC = {roc_auc:.4f}")
            ax.legend(loc="lower right", fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("ROC curves plot saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 4. Precision–Recall Curves
    # ------------------------------------------------------------------

    def plot_precision_recall_curves(
        self,
        per_class_pr: Dict[str, Dict[str, List[float]]],
    ) -> str:
        """Per-class Precision–Recall curves.

        Parameters
        ----------
        per_class_pr : dict  {class_name: {"precision": [...], "recall": [...]}}

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "precision_recall_curves.png")

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(10, 8))

            for i, (cls_name, curves) in enumerate(per_class_pr.items()):
                colour = _PALETTE[i % len(_PALETTE)]
                ax.plot(curves["recall"], curves["precision"],
                        color=colour, linewidth=1.5, alpha=0.8,
                        label=cls_name)

            ax.set_xlim([-0.01, 1.01])
            ax.set_ylim([-0.01, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision–Recall Curves (One-vs-Rest)")
            ax.legend(loc="upper right", fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Precision–Recall curves plot saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # 5. Per-class Performance Chart
    # ------------------------------------------------------------------

    def plot_per_class_performance(
        self,
        classification_report: Dict,
        class_names: List[str],
    ) -> str:
        """Grouped bar chart: precision, recall, F1 per class.

        Parameters
        ----------
        classification_report : dict  (from sklearn)
        class_names : list of str

        Returns
        -------
        str : Path to saved plot.
        """
        path = str(self.output_dir / "per_class_performance.png")

        precisions = []
        recalls = []
        f1s = []
        for cls in class_names:
            entry = classification_report.get(cls, {})
            precisions.append(entry.get("precision", 0.0))
            recalls.append(entry.get("recall", 0.0))
            f1s.append(entry.get("f1-score", 0.0))

        x = np.arange(len(class_names))
        width = 0.25

        with plt.rc_context(_STYLE):
            fig, ax = plt.subplots(figsize=(14, 7))

            ax.bar(x - width, precisions, width, label="Precision", color="#58a6ff", alpha=0.85)
            ax.bar(x,         recalls,    width, label="Recall",    color="#f78166", alpha=0.85)
            ax.bar(x + width, f1s,        width, label="F1-Score",  color="#3fb950", alpha=0.85)

            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Score")
            ax.set_ylim([0, 1.05])
            ax.set_title("Per-Class Performance — XGBoost Attack Classifier")
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=DPI, bbox_inches="tight")
            plt.close(fig)

        logger.info("Per-class performance plot saved → %s", path)
        return path

    # ------------------------------------------------------------------
    # Convenience: plot everything at once
    # ------------------------------------------------------------------

    def plot_all(
        self,
        importance_df: pd.DataFrame,
        confusion_matrix: List[List[int]],
        class_names: List[str],
        per_class_roc: Dict,
        per_class_pr: Dict,
        roc_auc: float,
        classification_report: Dict,
    ) -> List[str]:
        """Generate all plots.

        Returns
        -------
        list of str – saved file paths.
        """
        paths = []
        paths.append(self.plot_feature_importance(importance_df))
        paths.append(self.plot_confusion_matrix(confusion_matrix, class_names))
        paths.append(self.plot_roc_curves(per_class_roc, roc_auc))
        paths.append(self.plot_precision_recall_curves(per_class_pr))
        paths.append(self.plot_per_class_performance(classification_report, class_names))
        logger.info("All %d plots generated.", len(paths))
        return paths
