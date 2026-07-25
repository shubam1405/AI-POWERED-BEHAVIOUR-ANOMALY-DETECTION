"""
global_explainer.py – Global SHAP feature importance for the full model.

Computes
--------
* Mean absolute SHAP values across all samples and all classes
  → overall model feature ranking.
* Per-class mean absolute SHAP
  → which features drive each attack type?

Saves
-----
* ``outputs/global_feature_importance.csv``
* ``outputs/per_class_feature_importance.csv``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from explainability.utils import ensure_dir

logger = logging.getLogger("Explainability.GlobalExplainer")


class GlobalExplainer:
    """Derive global SHAP feature importance from the full SHAP tensor.

    Parameters
    ----------
    shap_values : ndarray of shape ``(n_samples, n_features, n_classes)``
    feature_names : list of str
    class_names : list of str
    output_dir : str
    """

    def __init__(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        output_dir: str = "outputs",
    ) -> None:
        self.shap_values = shap_values          # (n_samples, n_features, n_classes)
        self.feature_names = feature_names
        self.class_names = class_names
        self.output_dir = output_dir

        self._global_df: Optional[pd.DataFrame] = None
        self._per_class_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> "GlobalExplainer":
        """Compute global and per-class importance. Returns self."""
        self._compute_global()
        self._compute_per_class()
        return self

    @property
    def global_importance(self) -> pd.DataFrame:
        """DataFrame: feature, mean_abs_shap, rank."""
        if self._global_df is None:
            raise RuntimeError("Call compute() first.")
        return self._global_df

    @property
    def per_class_importance(self) -> pd.DataFrame:
        """DataFrame: feature + one column per class."""
        if self._per_class_df is None:
            raise RuntimeError("Call compute() first.")
        return self._per_class_df

    def top_n(self, n: int = 20) -> pd.DataFrame:
        """Return top-N features by global mean |SHAP|."""
        return self.global_importance.head(n)

    def save(self) -> Dict[str, str]:
        """Save both CSVs.  Returns dict of {name: path}."""
        ensure_dir(self.output_dir)
        paths: Dict[str, str] = {}

        # Global
        global_path = str(Path(self.output_dir) / "global_feature_importance.csv")
        self.global_importance.to_csv(global_path, index=False)
        logger.info("Global feature importance saved → %s", global_path)
        paths["global"] = global_path

        # Per-class
        per_class_path = str(Path(self.output_dir) / "per_class_feature_importance.csv")
        self.per_class_importance.to_csv(per_class_path, index=False)
        logger.info("Per-class feature importance saved → %s", per_class_path)
        paths["per_class"] = per_class_path

        return paths

    def log_top_features(self, n: int = 20) -> None:
        """Log the top-N features to the standard logger."""
        top = self.top_n(n)
        logger.info("Top %d global features (mean |SHAP|):", n)
        for _, row in top.iterrows():
            logger.info(
                "  #%-3d  %-40s  mean_abs_shap=%.4f",
                int(row["rank"]), row["feature"], row["mean_abs_shap"],
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_global(self) -> None:
        # Mean over samples and classes: (n_features,)
        mean_abs = np.mean(np.abs(self.shap_values), axis=(0, 2))
        ranks = np.argsort(mean_abs)[::-1]

        self._global_df = pd.DataFrame({
            "feature":       [self.feature_names[i] for i in ranks],
            "mean_abs_shap": mean_abs[ranks].round(6),
            "rank":          np.arange(1, len(ranks) + 1),
        })

    def _compute_per_class(self) -> None:
        # Mean |SHAP| per feature per class: (n_features, n_classes)
        mean_per_class = np.mean(np.abs(self.shap_values), axis=0)   # (n_features, n_classes)

        data: Dict[str, List] = {"feature": self.feature_names}
        for cls_idx, cls_name in enumerate(self.class_names):
            data[cls_name] = mean_per_class[:, cls_idx].round(6).tolist()

        df = pd.DataFrame(data)
        # Sort by global mean_abs_shap (already computed)
        if self._global_df is not None:
            ordered_features = self._global_df["feature"].tolist()
            df["_sort_key"] = df["feature"].map(
                {f: i for i, f in enumerate(ordered_features)}
            )
            df = df.sort_values("_sort_key").drop(columns=["_sort_key"])

        self._per_class_df = df.reset_index(drop=True)
