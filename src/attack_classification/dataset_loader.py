"""
dataset_loader.py – Data loading and label preparation for Attack Classification.

Loads ``tabular_features.csv`` and ``reconstruction_scores.csv``, merges them
on ``session_id``, validates alignment, and returns a labelled dataset
ready for XGBoost training.

Split strategy
--------------
* Stratified 70 / 15 / 15 train / val / test split.
* All classes (Normal + attack types) are present in every split.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from attack_classification.utils import (
    ATTACK_LABELS,
    META_COLUMNS,
    SCORE_MERGE_COLUMNS,
    ensure_dir,
)

logger = logging.getLogger("AttackClassification.DatasetLoader")


class AttackDatasetLoader:
    """Load, merge, validate, and split data for attack classification.

    Parameters
    ----------
    tabular_path : str
        Path to ``tabular_features.csv``.
    scores_path : str
        Path to ``reconstruction_scores.csv`` (from GRU Autoencoder).
    train_ratio : float
        Fraction for training (default 0.70).
    val_ratio : float
        Fraction for validation (default 0.15).
    seed : int
        Random seed for reproducible splits.
    """

    def __init__(
        self,
        tabular_path: str = "data/processed/tabular_features.csv",
        scores_path: str = "outputs/reconstruction_scores.csv",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.tabular_path = Path(tabular_path)
        self.scores_path = Path(scores_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed

        # Populated after load()
        self.df: Optional[pd.DataFrame] = None
        self.feature_names: List[str] = []
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.class_names: List[str] = []
        self.n_classes: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "AttackDatasetLoader":
        """Load and merge both input files.  Returns self (fluent API)."""
        self._validate_paths()

        # 1. Load tabular features
        logger.info("Loading tabular features from %s …", self.tabular_path)
        tab_df = pd.read_csv(self.tabular_path)
        logger.info("  Tabular: %d rows × %d columns.", *tab_df.shape)

        # 2. Load reconstruction scores
        logger.info("Loading reconstruction scores from %s …", self.scores_path)
        scores_df = pd.read_csv(self.scores_path)
        logger.info("  Scores: %d rows × %d columns.", *scores_df.shape)

        # 3. Merge on session_id (left join — keep all tabular sessions)
        scores_subset = scores_df[SCORE_MERGE_COLUMNS].copy()
        merged = tab_df.merge(scores_subset, on="session_id", how="left")

        # 4. Fill missing scores (sessions not in the GRU test set)
        for col in ["reconstruction_error", "anomaly_score"]:
            n_missing = int(merged[col].isna().sum())
            if n_missing > 0:
                logger.info(
                    "  Filling %d missing '%s' values with 0.0 "
                    "(normal training sessions — low expected error).",
                    n_missing, col,
                )
                merged[col] = merged[col].fillna(0.0)

        # 5. Normalise attack_type labels
        merged["attack_type"] = merged["attack_type"].fillna("None")
        merged["attack_type"] = merged["attack_type"].replace({"None": "Normal", "none": "Normal"})

        # 6. Remove any fully-empty rows
        before = len(merged)
        merged.dropna(how="all", inplace=True)
        if len(merged) < before:
            logger.warning("Dropped %d fully-empty rows.", before - len(merged))

        # 7. Handle remaining NaN / inf in feature columns
        feature_cols = [c for c in merged.columns if c not in META_COLUMNS]
        for col in feature_cols:
            if merged[col].dtype == object:
                # Encode string categoricals to integer
                merged[col] = merged[col].astype("category").cat.codes.astype(float)
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
        merged[feature_cols] = merged[feature_cols].fillna(0.0)
        merged[feature_cols] = merged[feature_cols].replace(
            [np.inf, -np.inf], 0.0
        )

        # 8. Fit label encoder
        self.label_encoder.fit(sorted(merged["attack_type"].unique()))
        self.class_names = list(self.label_encoder.classes_)
        self.n_classes = len(self.class_names)
        merged["label"] = self.label_encoder.transform(merged["attack_type"])

        self.feature_names = feature_cols
        self.df = merged

        # Log class distribution
        dist = merged["attack_type"].value_counts()
        logger.info("Class distribution (%d classes):", self.n_classes)
        for cls_name in self.class_names:
            count = int(dist.get(cls_name, 0))
            logger.info("  %-30s %5d", cls_name, count)
        logger.info("Total features: %d", len(self.feature_names))

        return self

    def get_feature_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(X, y)`` numpy arrays for the full dataset.

        Returns
        -------
        X : ndarray of shape ``(n_samples, n_features)``
        y : ndarray of shape ``(n_samples,)`` – integer-encoded labels
        """
        if self.df is None:
            raise RuntimeError("Call load() first.")
        X = self.df[self.feature_names].values.astype(np.float32)
        y = self.df["label"].values.astype(np.int32)
        return X, y

    def get_session_ids(self) -> np.ndarray:
        """Return session IDs in the same order as the feature matrix."""
        if self.df is None:
            raise RuntimeError("Call load() first.")
        return self.df["session_id"].values

    def get_metadata(self) -> pd.DataFrame:
        """Return the full merged DataFrame (for export)."""
        if self.df is None:
            raise RuntimeError("Call load() first.")
        return self.df

    def split(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Stratified train / val / test split.

        Returns
        -------
        (X_train, y_train), (X_val, y_val), (X_test, y_test),
        (idx_train, idx_val, idx_test)
        """
        X, y = self.get_feature_matrix()
        indices = np.arange(len(y))

        test_ratio = 1.0 - self.train_ratio - self.val_ratio

        # First split: train+val vs test
        idx_tv, idx_test, y_tv, y_test = train_test_split(
            indices, y,
            test_size=test_ratio,
            stratify=y,
            random_state=self.seed,
        )

        # Second split: train vs val
        relative_val = self.val_ratio / (self.train_ratio + self.val_ratio)
        idx_train, idx_val, y_train, y_val = train_test_split(
            idx_tv, y_tv,
            test_size=relative_val,
            stratify=y_tv,
            random_state=self.seed,
        )

        logger.info(
            "Split → train=%d  val=%d  test=%d",
            len(idx_train), len(idx_val), len(idx_test),
        )

        return (
            (X[idx_train], y_train),
            (X[idx_val],   y_val),
            (X[idx_test],  y_test),
            (idx_train, idx_val, idx_test),
        )

    def save_merged(self, output_path: str = "outputs/merged_dataset.csv") -> str:
        """Export the merged dataset to CSV."""
        if self.df is None:
            raise RuntimeError("Call load() first.")
        ensure_dir(Path(output_path).parent)
        self.df.to_csv(output_path, index=False)
        logger.info("Merged dataset saved → %s  (%d rows)", output_path, len(self.df))
        return output_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        for p in (self.tabular_path, self.scores_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")
