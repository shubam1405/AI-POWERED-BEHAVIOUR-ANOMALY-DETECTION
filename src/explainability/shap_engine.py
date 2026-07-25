"""
shap_engine.py – Core SHAP computation engine for the Cyber Cage XAI layer.

Responsibilities
----------------
* Load the trained XGBoost model (pickle).
* Load ``merged_dataset.csv`` and ``attack_predictions.csv``; inner-join on
  ``session_id`` to attach ground truth labels, predictions, confidence,
  anomaly_score, and risk_score to every row.
* Identify the exact feature columns used during training (mirrors
  ``AttackDatasetLoader`` logic — excludes META_COLUMNS).
* Initialise ``shap.TreeExplainer`` (exact, fast for XGBoost).
* Compute SHAP values: shape ``(n_samples, n_features, n_classes)``.
* Cache SHAP values to ``outputs/shap_values.npz`` for fast re-use.
* Provide per-session slice helpers.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from explainability.utils import ensure_dir, elapsed_str

logger = logging.getLogger("Explainability.SHAPEngine")

# Columns that are NOT model features (must match attack_classification/utils.py)
_META_COLUMNS: List[str] = [
    "session_id",
    "employee_id",
    "is_anomalous",
    "attack_type",
    "risk_score",
    "label",
]

# Columns pulled from attack_predictions.csv
_PRED_COLUMNS: List[str] = [
    "session_id",
    "predicted_label",
    "confidence",
    "true_label",
    "top_1", "top_1_prob",
    "top_2", "top_2_prob",
    "top_3", "top_3_prob",
]


class SHAPEngine:
    """Load model and data; compute + cache multiclass SHAP values.

    Parameters
    ----------
    model_path : str
        Path to ``xgboost_attack_classifier.pkl``.
    dataset_path : str
        Path to ``merged_dataset.csv``.
    predictions_path : str
        Path to ``attack_predictions.csv``.
    cache_path : str
        Where to save/load the SHAP value cache (``shap_values.npz``).
    force_recompute : bool
        If *True*, ignore any existing cache and recompute.
    """

    def __init__(
        self,
        model_path: str = "models/xgboost_attack_classifier.pkl",
        dataset_path: str = "outputs/merged_dataset.csv",
        predictions_path: str = "outputs/attack_predictions.csv",
        cache_path: str = "outputs/shap_values.npz",
        force_recompute: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.predictions_path = Path(predictions_path)
        self.cache_path = Path(cache_path)
        self.force_recompute = force_recompute

        # Populated after load()
        self.model: Optional[xgb.XGBClassifier] = None
        self.df: Optional[pd.DataFrame] = None          # merged + predictions
        self.X: Optional[np.ndarray] = None             # (n_samples, n_features)
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.n_classes: int = 0

        # SHAP: (n_samples, n_features, n_classes) or list of arrays
        self.shap_values: Optional[np.ndarray] = None
        self.explainer: Optional[shap.TreeExplainer] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "SHAPEngine":
        """Load model + data. Returns self (fluent API)."""
        self._load_model()
        self._load_data()
        return self

    def compute_shap(self) -> "SHAPEngine":
        """Compute (or load cached) SHAP values.

        SHAP values have shape ``(n_samples, n_features, n_classes)``.
        """
        if not self.force_recompute and self.cache_path.exists():
            logger.info("Loading cached SHAP values from %s …", self.cache_path)
            data = np.load(self.cache_path)
            self.shap_values = data["shap_values"]
            logger.info(
                "  Loaded: shape=%s  classes=%d",
                self.shap_values.shape, self.n_classes,
            )
            return self

        logger.info(
            "Computing SHAP values for %d samples × %d features × %d classes …",
            len(self.X), len(self.feature_names), self.n_classes,
        )
        t0 = time.time()

        self.explainer = shap.TreeExplainer(
            self.model,
            feature_perturbation="interventional",
            model_output="raw",
        )

        # shap_values returns list[ndarray] of length n_classes for multiclass,
        # each of shape (n_samples, n_features).  Stack to 3-D.
        raw = self.explainer.shap_values(self.X)

        if isinstance(raw, list):
            # list of (n_samples, n_features) → (n_samples, n_features, n_classes)
            self.shap_values = np.stack(raw, axis=2)
        else:
            # Already (n_samples, n_features, n_classes)
            self.shap_values = raw

        logger.info(
            "SHAP computation complete in %s. Shape: %s",
            elapsed_str(t0), self.shap_values.shape,
        )

        # Cache
        ensure_dir(self.cache_path.parent)
        np.savez_compressed(self.cache_path, shap_values=self.shap_values)
        logger.info("SHAP values cached → %s", self.cache_path)

        return self

    def get_shap_for_session(
        self,
        session_id: str,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Return SHAP values and feature vector for one session.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        shap_matrix : ndarray of shape ``(n_features, n_classes)``
        feature_vector : ndarray of shape ``(n_features,)``
        meta : dict with session metadata (prediction, confidence, etc.)
        """
        if self.shap_values is None:
            raise RuntimeError("Call compute_shap() first.")
        idx = self._get_session_index(session_id)
        shap_matrix = self.shap_values[idx]          # (n_features, n_classes)
        feature_vector = self.X[idx]                  # (n_features,)
        row = self.df.iloc[idx]
        meta = {
            "session_id":      session_id,
            "predicted_label": str(row.get("predicted_label", "")),
            "true_label":      str(row.get("true_label",      str(row.get("attack_type", "")))),
            "confidence":      float(row.get("confidence",    0.0)),
            "anomaly_score":   float(row.get("anomaly_score", 0.0)),
            "risk_score":      float(row.get("risk_score",    0.0)),
        }
        return shap_matrix, feature_vector, meta

    def get_predicted_class_index(self, session_id: str) -> int:
        """Return integer class index for the predicted label of *session_id*."""
        idx = self._get_session_index(session_id)
        label = str(self.df.iloc[idx].get("predicted_label", "Normal"))
        try:
            return self.class_names.index(label)
        except ValueError:
            return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info("Model loaded ← %s", self.model_path)

        # Derive class names from model if available
        if hasattr(self.model, "classes_"):
            self.class_names = [str(c) for c in self.model.classes_]
            self.n_classes = len(self.class_names)

    def _load_data(self) -> None:
        # Load merged dataset
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        df = pd.read_csv(self.dataset_path)
        logger.info("Dataset loaded: %d rows × %d cols", *df.shape)

        # Load predictions
        if not self.predictions_path.exists():
            raise FileNotFoundError(f"Predictions not found: {self.predictions_path}")
        preds_df = pd.read_csv(self.predictions_path)
        logger.info("Predictions loaded: %d rows", len(preds_df))

        # Merge predictions onto dataset (left join — keep all dataset rows)
        avail_pred_cols = [c for c in _PRED_COLUMNS if c in preds_df.columns]
        df = df.merge(preds_df[avail_pred_cols], on="session_id", how="left")

        # If predictions don't cover all rows, fall back to attack_type
        if "predicted_label" in df.columns:
            df["predicted_label"] = df["predicted_label"].fillna(
                df.get("attack_type", "Normal")
            )
        else:
            df["predicted_label"] = df.get("attack_type", "Normal")

        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].fillna(0.5)
        else:
            df["confidence"] = 0.5

        # Identify feature columns
        excluded = set(_META_COLUMNS)
        for col in avail_pred_cols:
            excluded.add(col)
        self.feature_names = [
            c for c in df.columns
            if c not in excluded and df[c].dtype != object
        ]

        # Clean feature matrix
        X_df = df[self.feature_names].copy()
        X_df = X_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self.X = X_df.values.astype(np.float32)

        # Derive class names from data if not already set
        if not self.class_names and "predicted_label" in df.columns:
            self.class_names = sorted(df["predicted_label"].dropna().unique().tolist())
            self.n_classes = len(self.class_names)

        self.df = df.reset_index(drop=True)
        logger.info(
            "Data ready: %d samples, %d features, %d classes.",
            len(self.df), len(self.feature_names), self.n_classes,
        )

    def _get_session_index(self, session_id: str) -> int:
        matches = self.df.index[self.df["session_id"] == session_id].tolist()
        if not matches:
            raise KeyError(f"session_id not found: {session_id!r}")
        return int(matches[0])
