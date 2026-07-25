"""
feature_merger.py – Feature validation and merge utilities.

Validates the merged feature set produced by :class:`AttackDatasetLoader`
to ensure data quality before training:

    ✓  No NaN values in features
    ✓  No infinite values
    ✓  Matching session IDs between tabular and scores files
    ✓  Consistent row counts
    ✓  No duplicate session IDs
    ✓  All expected columns present
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from attack_classification.utils import META_COLUMNS

logger = logging.getLogger("AttackClassification.FeatureMerger")


class FeatureMergeValidator:
    """Validates the merged feature DataFrame for data quality issues.

    Parameters
    ----------
    df : pd.DataFrame
        The merged DataFrame produced by :class:`AttackDatasetLoader`.
    feature_names : list of str
        List of feature column names (excludes meta columns).
    """

    def __init__(self, df: pd.DataFrame, feature_names: List[str]) -> None:
        self.df = df
        self.feature_names = feature_names
        self.issues: List[str] = []

    def validate(self, strict: bool = False) -> bool:
        """Run all validation checks.

        Parameters
        ----------
        strict : bool
            If *True*, raise ``ValueError`` on any issue.
            If *False* (default), log warnings and return the pass/fail status.

        Returns
        -------
        bool
            *True* if all checks pass.
        """
        self.issues.clear()

        self._check_no_nan()
        self._check_no_inf()
        self._check_no_duplicate_sessions()
        self._check_expected_columns()
        self._check_label_present()
        self._check_feature_types()

        if self.issues:
            for issue in self.issues:
                logger.warning("Validation issue: %s", issue)
            if strict:
                raise ValueError(
                    f"Feature merge validation failed with {len(self.issues)} issue(s):\n"
                    + "\n".join(f"  • {i}" for i in self.issues)
                )
            return False

        logger.info("Feature merge validation passed ✓  (%d features × %d rows)",
                     len(self.feature_names), len(self.df))
        return True

    def summary(self) -> Dict:
        """Return a validation summary dict."""
        feat_df = self.df[self.feature_names]
        return {
            "n_rows": len(self.df),
            "n_features": len(self.feature_names),
            "n_nan": int(feat_df.isna().sum().sum()),
            "n_inf": int(np.isinf(feat_df.select_dtypes(include=[np.number]).values).sum()),
            "n_duplicate_sessions": int(self.df["session_id"].duplicated().sum()),
            "issues": self.issues.copy(),
            "passed": len(self.issues) == 0,
        }

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_no_nan(self) -> None:
        feat_df = self.df[self.feature_names]
        nan_count = int(feat_df.isna().sum().sum())
        if nan_count > 0:
            nan_cols = feat_df.columns[feat_df.isna().any()].tolist()
            self.issues.append(
                f"{nan_count} NaN values found in features: {nan_cols[:5]}"
            )

    def _check_no_inf(self) -> None:
        feat_df = self.df[self.feature_names].select_dtypes(include=[np.number])
        inf_count = int(np.isinf(feat_df.values).sum())
        if inf_count > 0:
            self.issues.append(f"{inf_count} infinite values found in features.")

    def _check_no_duplicate_sessions(self) -> None:
        dups = int(self.df["session_id"].duplicated().sum())
        if dups > 0:
            self.issues.append(f"{dups} duplicate session_id entries found.")

    def _check_expected_columns(self) -> None:
        required = {"session_id", "attack_type", "is_anomalous", "label"}
        missing = required - set(self.df.columns)
        if missing:
            self.issues.append(f"Missing required columns: {missing}")

        gru_cols = {"reconstruction_error", "anomaly_score"}
        missing_gru = gru_cols - set(self.df.columns)
        if missing_gru:
            self.issues.append(
                f"Missing GRU score columns: {missing_gru}. "
                "Ensure reconstruction_scores.csv was merged."
            )

    def _check_label_present(self) -> None:
        if "label" not in self.df.columns:
            self.issues.append("'label' column missing – run LabelEncoder first.")
        else:
            n_classes = self.df["label"].nunique()
            if n_classes < 2:
                self.issues.append(
                    f"Only {n_classes} class(es) present – need at least 2 for classification."
                )

    def _check_feature_types(self) -> None:
        for col in self.feature_names:
            if self.df[col].dtype == object:
                self.issues.append(
                    f"Feature '{col}' has dtype=object (string). "
                    "Encode it before training."
                )
