"""
inference.py – Real-time attack classification for new sessions.

Given a feature vector (tabular features + GRU anomaly score), the
:class:`AttackInferenceEngine` predicts the attack type with class
probabilities and a confidence score.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xgboost as xgb

from attack_classification.utils import ATTACK_LABELS

logger = logging.getLogger("AttackClassification.Inference")


class AttackInferenceEngine:
    """Scores new sessions using a trained XGBoost attack classifier.

    Parameters
    ----------
    model_path : str
        Path to the saved ``xgboost_attack_classifier.pkl``.
    class_names : list of str, optional
        Ordered class labels.  Defaults to :data:`ATTACK_LABELS`.
    feature_names : list of str, optional
        Ordered feature names for explainability.

    Example
    -------
    >>> engine = AttackInferenceEngine("models/xgboost_attack_classifier.pkl")
    >>> result = engine.predict(session_id="S-001", features=feature_vector)
    >>> print(result["prediction"])   # e.g. "Data Exfiltration"
    """

    def __init__(
        self,
        model_path: str = "models/xgboost_attack_classifier.pkl",
        class_names: Optional[List[str]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.model_path = model_path
        self.class_names = class_names or ATTACK_LABELS
        self.feature_names = feature_names or []

        logger.info("Loading attack classifier from %s …", model_path)
        with open(model_path, "rb") as f:
            self.model: xgb.XGBClassifier = pickle.load(f)
        logger.info("Attack classifier loaded (%d classes).", len(self.class_names))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        session_id: str,
        features: np.ndarray,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Classify a single session.

        Parameters
        ----------
        session_id : str
        features : ndarray of shape ``(n_features,)``
            Feature vector (must match training feature order).
        top_k : int
            Number of top predictions to return.

        Returns
        -------
        dict with keys:
            ``session_id``, ``prediction``, ``confidence``, ``top_predictions``
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        proba = self.model.predict_proba(features)[0]
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])

        # Top-k predictions
        top_indices = np.argsort(proba)[::-1][:top_k]
        top_predictions = [
            {
                "attack": self.class_names[idx],
                "probability": round(float(proba[idx]), 4),
            }
            for idx in top_indices
        ]

        result = {
            "session_id":      session_id,
            "prediction":      self.class_names[pred_idx],
            "confidence":      round(confidence, 4),
            "top_predictions": top_predictions,
        }

        logger.debug(
            "Session %s → %s (conf=%.4f)",
            session_id, result["prediction"], confidence,
        )
        return result

    def predict_batch(
        self,
        session_ids: List[str],
        features: np.ndarray,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Classify multiple sessions.

        Parameters
        ----------
        session_ids : list of str
        features : ndarray of shape ``(n_samples, n_features)``
        top_k : int

        Returns
        -------
        list of result dicts.
        """
        results = []
        proba_all = self.model.predict_proba(features)

        for i, sid in enumerate(session_ids):
            proba = proba_all[i]
            pred_idx = int(np.argmax(proba))
            confidence = float(proba[pred_idx])

            top_indices = np.argsort(proba)[::-1][:top_k]
            top_predictions = [
                {
                    "attack": self.class_names[idx],
                    "probability": round(float(proba[idx]), 4),
                }
                for idx in top_indices
            ]

            results.append({
                "session_id":      sid,
                "prediction":      self.class_names[pred_idx],
                "confidence":      round(confidence, 4),
                "top_predictions": top_predictions,
            })

        logger.info("Batch prediction complete: %d sessions.", len(results))
        return results

    def get_model(self) -> xgb.XGBClassifier:
        """Return the underlying XGBoost model (for SHAP integration)."""
        return self.model

    def get_feature_names(self) -> List[str]:
        """Return the feature names (for SHAP integration)."""
        return self.feature_names
