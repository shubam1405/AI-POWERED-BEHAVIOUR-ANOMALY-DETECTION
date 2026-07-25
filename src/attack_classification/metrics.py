"""
metrics.py – Feature importance extraction and export.

Extracts three types of XGBoost feature importance:
    - **Weight** (frequency): number of times a feature is used to split.
    - **Gain**: average gain in loss reduction when a feature is used.
    - **Cover**: average number of samples affected by a feature.

Exports a ranked CSV and returns the Top-20 features for visualization.
"""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from attack_classification.utils import ensure_dir

logger = logging.getLogger("AttackClassification.Metrics")


class FeatureImportanceAnalyzer:
    """Extracts and exports XGBoost feature importance.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained model.
    feature_names : list of str
        Ordered feature column names.
    output_dir : str
        Where to save outputs.
    """

    def __init__(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str],
        output_dir: str = "outputs",
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self.output_dir = output_dir

    def compute(self) -> pd.DataFrame:
        """Compute all three importance types and return a combined DataFrame.

        Returns
        -------
        pd.DataFrame with columns:
            ``feature``, ``weight``, ``gain``, ``cover``, ``rank_gain``
        """
        booster = self.model.get_booster()

        weight_scores = booster.get_score(importance_type="weight")
        gain_scores   = booster.get_score(importance_type="gain")
        cover_scores  = booster.get_score(importance_type="cover")

        rows = []
        for i, name in enumerate(self.feature_names):
            key = f"f{i}"
            rows.append({
                "feature": name,
                "weight":  weight_scores.get(key, 0.0),
                "gain":    gain_scores.get(key, 0.0),
                "cover":   cover_scores.get(key, 0.0),
            })

        df = pd.DataFrame(rows)
        df.sort_values("gain", ascending=False, inplace=True)
        df["rank_gain"] = range(1, len(df) + 1)
        df.reset_index(drop=True, inplace=True)

        logger.info("Feature importance computed for %d features.", len(df))
        return df

    def save(
        self,
        df: Optional[pd.DataFrame] = None,
        output_path: str = "outputs/feature_importance.csv",
    ) -> str:
        """Save feature importance to CSV.

        Parameters
        ----------
        df : pd.DataFrame, optional
            If not provided, calls :meth:`compute` first.
        output_path : str

        Returns
        -------
        str : Path to saved CSV.
        """
        if df is None:
            df = self.compute()
        ensure_dir(Path(output_path).parent)
        df.to_csv(output_path, index=False)
        logger.info("Feature importance saved → %s", output_path)
        return output_path

    def top_n(self, n: int = 20, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Return the top-N features by gain importance.

        Parameters
        ----------
        n : int
        df : pd.DataFrame, optional

        Returns
        -------
        pd.DataFrame
        """
        if df is None:
            df = self.compute()
        top = df.head(n)
        logger.info("Top %d features by gain:", n)
        for _, row in top.iterrows():
            logger.info("  #%d  %-35s  gain=%.2f", int(row["rank_gain"]), row["feature"], row["gain"])
        return top
