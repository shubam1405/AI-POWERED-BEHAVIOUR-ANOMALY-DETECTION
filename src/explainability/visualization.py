"""
visualization.py – SHAP visualizations for the Cyber Cage XAI Engine.

Generates 6 plot types, all saved to ``outputs/plots/`` with a consistent
dark theme (background ``#0d1117``).

Plots produced
--------------
1. shap_summary.png       – dot summary (global)
2. shap_bar.png           – mean |SHAP| bar (global)
3. shap_beeswarm.png      – beeswarm (global)
4. waterfall_example.png  – waterfall for most anomalous session (local)
5. dependence_plot_<feat>.png × 2  – dependence for top-2 features (global)
6. force_plot.html        – interactive force plot (optional, local)
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from explainability.utils import ensure_dir

logger = logging.getLogger("Explainability.Visualization")

# ---------------------------------------------------------------------------
# Dark theme constants
# ---------------------------------------------------------------------------
_BG   = "#0d1117"
_FG   = "#e6edf3"
_GRID = "#21262d"
_ACC1 = "#58a6ff"
_ACC2 = "#f78166"
_FONT = "DejaVu Sans"

plt.rcParams.update({
    "figure.facecolor":  _BG,
    "axes.facecolor":    _BG,
    "axes.edgecolor":    _GRID,
    "axes.labelcolor":   _FG,
    "xtick.color":       _FG,
    "ytick.color":       _FG,
    "text.color":        _FG,
    "grid.color":        _GRID,
    "font.family":       _FONT,
    "savefig.facecolor": _BG,
    "savefig.edgecolor": _BG,
})

warnings.filterwarnings("ignore", category=FutureWarning)


class ExplainabilityVisualizer:
    """Generate all SHAP visualizations.

    Parameters
    ----------
    shap_values : ndarray of shape ``(n_samples, n_features, n_classes)``
    X : ndarray of shape ``(n_samples, n_features)``
    feature_names : list of str
    class_names : list of str
    output_dir : str
    explainer : shap.TreeExplainer, optional
        Required for force plot generation.
    """

    def __init__(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        output_dir: str = "outputs/plots",
        explainer: Optional[shap.TreeExplainer] = None,
    ) -> None:
        self.shap_values  = shap_values       # (n, f, c)
        self.X            = X                 # (n, f)
        self.feature_names = feature_names
        self.class_names  = class_names
        self.output_dir   = output_dir
        self.explainer    = explainer
        ensure_dir(output_dir)

        # Mean-over-classes SHAP: (n_samples, n_features) — used for global plots
        self._shap_mean_cls = np.mean(self.shap_values, axis=2)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def plot_all(
        self,
        top_n_features: int = 20,
        skip_force: bool = False,
    ) -> List[str]:
        """Generate all plots.  Returns list of saved file paths."""
        paths: List[str] = []
        paths.extend(self._plot_summary(top_n_features))
        paths.extend(self._plot_bar(top_n_features))
        paths.extend(self._plot_beeswarm(top_n_features))
        paths.extend(self._plot_waterfall())
        paths.extend(self._plot_dependence(top_n_features))
        if not skip_force and self.explainer is not None:
            fp = self._plot_force()
            if fp:
                paths.append(fp)

        logger.info("All %d SHAP plots saved to %s/", len(paths), self.output_dir)
        return paths

    # ------------------------------------------------------------------
    # Individual plot methods
    # ------------------------------------------------------------------

    def _plot_summary(self, top_n: int) -> List[str]:
        """SHAP dot summary plot."""
        path = os.path.join(self.output_dir, "shap_summary.png")
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                self._shap_mean_cls,
                self.X,
                feature_names=self.feature_names,
                max_display=top_n,
                show=False,
                plot_type="dot",
                color_bar=True,
            )
            plt.title("SHAP Summary Plot — Feature Impact on Attack Prediction",
                      color=_FG, fontsize=13, pad=14)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("SHAP summary plot saved → %s", path)
        except Exception as e:
            logger.warning("Summary plot failed: %s", e)
            return []
        return [path]

    def _plot_bar(self, top_n: int) -> List[str]:
        """SHAP mean |value| bar chart."""
        path = os.path.join(self.output_dir, "shap_bar.png")
        try:
            shap.summary_plot(
                self._shap_mean_cls,
                self.X,
                feature_names=self.feature_names,
                max_display=top_n,
                show=False,
                plot_type="bar",
            )
            plt.title("Mean Absolute SHAP Values — Top Features",
                      color=_FG, fontsize=13, pad=14)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("SHAP bar plot saved → %s", path)
        except Exception as e:
            logger.warning("Bar plot failed: %s", e)
            return []
        return [path]

    def _plot_beeswarm(self, top_n: int) -> List[str]:
        """SHAP beeswarm plot using Explanation object."""
        path = os.path.join(self.output_dir, "shap_beeswarm.png")
        try:
            explanation = shap.Explanation(
                values=self._shap_mean_cls,
                data=self.X,
                feature_names=self.feature_names,
            )
            shap.plots.beeswarm(explanation, max_display=top_n, show=False)
            plt.title("SHAP Beeswarm Plot — Feature Value vs. Impact",
                      color=_FG, fontsize=13, pad=14)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("SHAP beeswarm plot saved → %s", path)
        except Exception as e:
            logger.warning("Beeswarm plot failed: %s", e)
            return []
        return [path]

    def _plot_waterfall(self) -> List[str]:
        """Waterfall plot for the most anomalous session."""
        path = os.path.join(self.output_dir, "waterfall_example.png")
        try:
            # Pick the session with highest mean |SHAP| — most "interesting"
            sample_importance = np.mean(np.abs(self._shap_mean_cls), axis=1)
            top_idx = int(np.argmax(sample_importance))

            shap_vals_1d = self._shap_mean_cls[top_idx]
            x_vals_1d    = self.X[top_idx]
            base_val     = float(np.mean(self._shap_mean_cls))

            explanation = shap.Explanation(
                values=shap_vals_1d,
                base_values=base_val,
                data=x_vals_1d,
                feature_names=self.feature_names,
            )
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.title("SHAP Waterfall — Most Anomalous Session",
                      color=_FG, fontsize=13, pad=14)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close("all")
            logger.info("Waterfall plot saved → %s", path)
        except Exception as e:
            logger.warning("Waterfall plot failed: %s", e)
            return []
        return [path]

    def _plot_dependence(self, top_n: int = 20) -> List[str]:
        """Dependence plots for the top-2 most important features."""
        # Rank features by mean |SHAP|
        mean_abs = np.mean(np.abs(self._shap_mean_cls), axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:2]

        paths: List[str] = []
        for fi in top_indices:
            fname = self.feature_names[fi]
            safe_name = fname.replace(" ", "_").replace("/", "_")
            path = os.path.join(self.output_dir, f"dependence_plot_{safe_name}.png")
            try:
                shap.dependence_plot(
                    fi,
                    self._shap_mean_cls,
                    self.X,
                    feature_names=self.feature_names,
                    show=False,
                )
                plt.title(f"SHAP Dependence — {fname}",
                          color=_FG, fontsize=13, pad=14)
                plt.tight_layout()
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close("all")
                logger.info("Dependence plot saved → %s", path)
                paths.append(path)
            except Exception as e:
                logger.warning("Dependence plot for '%s' failed: %s", fname, e)
        return paths

    def _plot_force(self) -> Optional[str]:
        """Interactive HTML force plot for the most anomalous session."""
        path = os.path.join(self.output_dir, "force_plot.html")
        try:
            if self.explainer is None:
                return None
            sample_importance = np.mean(np.abs(self._shap_mean_cls), axis=1)
            top_idx = int(np.argmax(sample_importance))

            # For TreeExplainer the expected_value is per-class for multiclass
            ev = self.explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                ev = float(ev[0])

            fp = shap.force_plot(
                ev,
                self._shap_mean_cls[top_idx],
                self.X[top_idx],
                feature_names=self.feature_names,
                show=False,
                matplotlib=False,
            )
            shap.save_html(path, fp)
            logger.info("Force plot HTML saved → %s", path)
            return path
        except Exception as e:
            logger.warning("Force plot failed: %s", e)
            return None
