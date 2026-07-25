"""
local_explainer.py – Per-session SHAP explanation with enhanced metadata.

For each session produces a structured dict containing:

    * Prediction, confidence, anomaly score, risk score
    * Severity (Low / Medium / High / Critical)
    * MITRE ATT&CK mapping
    * Positive SHAP contributors (features that increased the prediction)
    * Negative SHAP contributors (features that decreased / opposed the prediction)
    * Feature category for every contributor
    * Recommended investigation steps
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from explainability.utils import (
    MITRE_MAPPING,
    INVESTIGATION_STEPS,
    compute_severity,
    get_feature_category,
    get_feature_description,
    get_impact_label,
    elapsed_str,
)

logger = logging.getLogger("Explainability.LocalExplainer")


class LocalExplainer:
    """Generate per-session structured explanation dicts.

    Parameters
    ----------
    shap_values : ndarray of shape ``(n_samples, n_features, n_classes)``
    X : ndarray of shape ``(n_samples, n_features)``
    feature_names : list of str
    class_names : list of str
    df : pd.DataFrame
        The merged dataset (must contain ``session_id``, ``predicted_label``,
        ``confidence``, ``anomaly_score``, ``risk_score``).
    top_n : int
        Number of top contributors per direction (positive / negative).
    """

    def __init__(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        feature_names: List[str],
        class_names: List[str],
        df,
        top_n: int = 10,
    ) -> None:
        self.shap_values = shap_values          # (n, f, c)
        self.X = X                              # (n, f)
        self.feature_names = feature_names
        self.class_names = class_names
        self.df = df.reset_index(drop=True)
        self.top_n = top_n

        # Build session_id → row index lookup
        self._session_index: Dict[str, int] = {
            str(sid): i
            for i, sid in enumerate(self.df["session_id"])
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain_session(self, session_id: str) -> Dict[str, Any]:
        """Generate a full explanation for a single session.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        dict  (see module docstring for structure)
        """
        idx = self._get_index(session_id)
        return self._build_explanation(idx)

    def explain_all(self, log_every: int = 500) -> List[Dict[str, Any]]:
        """Explain every session in the dataset.

        Parameters
        ----------
        log_every : int
            Log progress every N sessions.

        Returns
        -------
        list of dicts
        """
        n = len(self.df)
        t0 = time.time()
        explanations: List[Dict[str, Any]] = []

        for i in range(n):
            explanations.append(self._build_explanation(i))
            if (i + 1) % log_every == 0 or (i + 1) == n:
                logger.info(
                    "  Explained %d / %d sessions (%s elapsed)",
                    i + 1, n, elapsed_str(t0),
                )

        logger.info(
            "Local explanations complete: %d sessions in %s",
            n, elapsed_str(t0),
        )
        return explanations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_index(self, session_id: str) -> int:
        if session_id not in self._session_index:
            raise KeyError(f"session_id not found: {session_id!r}")
        return self._session_index[session_id]

    def _build_explanation(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        session_id = str(row["session_id"])

        # --- Metadata ---
        prediction   = str(row.get("predicted_label", "Normal"))
        confidence   = float(row.get("confidence",    0.5))
        anomaly_sc   = float(row.get("anomaly_score", 0.0))
        risk_sc      = float(row.get("risk_score",    0.0))
        true_label   = str(row.get("true_label",      row.get("attack_type", "")))

        # --- Severity ---
        severity = compute_severity(confidence, anomaly_sc, risk_sc)

        # --- MITRE mapping ---
        mitre = MITRE_MAPPING.get(prediction)

        # --- SHAP slice for predicted class ---
        cls_idx = self._class_index(prediction)
        shap_for_class = self.shap_values[idx, :, cls_idx]   # (n_features,)
        feature_values = self.X[idx]                          # (n_features,)

        # --- Split contributors ---
        positive_contributors, negative_contributors = self._split_contributors(
            shap_for_class, feature_values
        )

        # --- Investigation steps ---
        steps = INVESTIGATION_STEPS.get(prediction, INVESTIGATION_STEPS["Normal"])

        return {
            "session_id":              session_id,
            "prediction":              prediction,
            "true_label":              true_label,
            "confidence":              round(confidence, 4),
            "anomaly_score":           round(anomaly_sc, 4),
            "risk_score":              round(risk_sc, 4),
            "severity":                severity,
            "mitre":                   mitre,
            "positive_contributors":   positive_contributors,
            "negative_contributors":   negative_contributors,
            "investigation_steps":     steps,
        }

    def _split_contributors(
        self,
        shap_vals: np.ndarray,
        feat_vals: np.ndarray,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split features into positive and negative SHAP lists."""
        positive: List[Dict[str, Any]] = []
        negative: List[Dict[str, Any]] = []

        # Sort by absolute SHAP value descending
        ranked = np.argsort(np.abs(shap_vals))[::-1]

        pos_count = neg_count = 0
        for fi in ranked:
            sv = float(shap_vals[fi])
            fv = float(feat_vals[fi])
            fname = self.feature_names[fi]
            entry = {
                "feature":     fname,
                "description": get_feature_description(fname),
                "category":    get_feature_category(fname),
                "value":       round(fv, 4),
                "shap_value":  round(sv, 4),
                "impact":      get_impact_label(sv),
            }
            if sv > 0:
                if pos_count < self.top_n:
                    positive.append(entry)
                    pos_count += 1
            else:
                if neg_count < self.top_n:
                    negative.append(entry)
                    neg_count += 1

            if pos_count >= self.top_n and neg_count >= self.top_n:
                break

        return positive, negative

    def _class_index(self, class_name: str) -> int:
        try:
            return self.class_names.index(class_name)
        except ValueError:
            return 0
