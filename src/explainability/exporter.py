"""
exporter.py – Export session explanations and copilot-ready context.

Generates
---------
* ``outputs/session_explanations.json``  – full enriched objects (array)
* ``outputs/session_explanations.csv``   – flat tabular version
* A ``copilot_context`` field on every explanation dict for Phase 8.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from explainability.utils import ensure_dir

logger = logging.getLogger("Explainability.Exporter")

# Flat CSV columns (order matters for readability)
_CSV_META_COLS = [
    "session_id",
    "prediction",
    "true_label",
    "confidence",
    "anomaly_score",
    "risk_score",
    "severity",
]
_MAX_CSV_FEATURES = 5   # Top-N positive contributor feature names in flat CSV


class AttackExplainExporter:
    """Assemble copilot_context and export all explanation artefacts.

    Parameters
    ----------
    output_dir : str
        Parent directory for JSON and CSV outputs.
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir
        ensure_dir(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def attach_copilot_context(
        self,
        explanations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add a ``copilot_context`` field to every explanation dict in-place.

        The ``copilot_context`` is structured for direct consumption by the
        Phase 8 AI Security Copilot.

        Parameters
        ----------
        explanations : list of dicts
            Output from ``ExplanationGenerator.generate_batch()``.

        Returns
        -------
        list of dicts (mutated in-place, also returned)
        """
        for exp in explanations:
            mitre = exp.get("mitre") or {}
            pos   = exp.get("positive_contributors", [])
            top_features = [c["feature"] for c in pos[:5]]

            exp["copilot_context"] = {
                "attack_type":        exp.get("prediction",   "Unknown"),
                "confidence":         exp.get("confidence",   0.0),
                "severity":           exp.get("severity",     "Low"),
                "risk_score":         exp.get("risk_score",   0.0),
                "anomaly_score":      exp.get("anomaly_score",0.0),
                "mitre_tactic":       mitre.get("tactic",         ""),
                "mitre_technique_id": mitre.get("technique_id",   ""),
                "mitre_technique":    mitre.get("technique",      ""),
                "top_features":       top_features,
                "investigation_steps":exp.get("investigation_steps", []),
                "incident_summary":   exp.get("summary", ""),
            }

        logger.info(
            "copilot_context attached to %d session explanations.", len(explanations)
        )
        return explanations

    def save_json(
        self,
        explanations: List[Dict[str, Any]],
        filename: str = "session_explanations.json",
    ) -> str:
        """Save full explanations to JSON.

        Parameters
        ----------
        explanations : list of dicts
        filename : str

        Returns
        -------
        str : Absolute path of saved file.
        """
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(explanations, f, indent=2, ensure_ascii=False, default=str)
        logger.info(
            "Session explanations JSON saved → %s  (%d sessions)",
            path, len(explanations),
        )
        return path

    def save_csv(
        self,
        explanations: List[Dict[str, Any]],
        filename: str = "session_explanations.csv",
    ) -> str:
        """Save flat CSV version of explanations.

        Each row contains metadata + top positive feature names/values +
        the natural language summary.

        Returns
        -------
        str : Absolute path of saved file.
        """
        rows: List[Dict[str, Any]] = []
        for exp in explanations:
            row: Dict[str, Any] = {}

            # Metadata
            for col in _CSV_META_COLS:
                row[col] = exp.get(col, "")

            # MITRE
            mitre = exp.get("mitre") or {}
            row["mitre_tactic"]       = mitre.get("tactic", "")
            row["mitre_technique_id"] = mitre.get("technique_id", "")
            row["mitre_technique"]    = mitre.get("technique", "")

            # Top positive contributors (feature name + value)
            pos = exp.get("positive_contributors", [])
            for i in range(_MAX_CSV_FEATURES):
                if i < len(pos):
                    row[f"top_pos_feature_{i+1}"]   = pos[i].get("feature", "")
                    row[f"top_pos_value_{i+1}"]      = pos[i].get("value",   "")
                    row[f"top_pos_category_{i+1}"]   = pos[i].get("category","")
                    row[f"top_pos_impact_{i+1}"]     = pos[i].get("impact",  "")
                else:
                    row[f"top_pos_feature_{i+1}"]   = ""
                    row[f"top_pos_value_{i+1}"]      = ""
                    row[f"top_pos_category_{i+1}"]   = ""
                    row[f"top_pos_impact_{i+1}"]     = ""

            # Top negative contributors
            neg = exp.get("negative_contributors", [])
            for i in range(3):
                if i < len(neg):
                    row[f"top_neg_feature_{i+1}"]   = neg[i].get("feature", "")
                    row[f"top_neg_shap_{i+1}"]       = neg[i].get("shap_value", "")
                else:
                    row[f"top_neg_feature_{i+1}"]   = ""
                    row[f"top_neg_shap_{i+1}"]       = ""

            # Summaries
            row["nl_explanation"] = exp.get("nl_explanation", "")
            row["summary"]        = exp.get("summary",        "")

            rows.append(row)

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, filename)
        df.to_csv(path, index=False)
        logger.info(
            "Session explanations CSV saved → %s  (%d rows × %d cols)",
            path, len(df), len(df.columns),
        )
        return path
