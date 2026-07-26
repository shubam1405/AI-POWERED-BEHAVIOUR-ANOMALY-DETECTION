"""
context_builder.py – Build unified IncidentContext objects from Phase 7 outputs.

Loads ``session_explanations.json`` once (lazy, cached), then merges with
``attack_predictions.csv`` and ``merged_dataset.csv`` to produce complete
``IncidentContext`` dataclass instances ready for all copilot modules.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from copilot.utils import IncidentContext, elapsed_str

logger = logging.getLogger("Copilot.ContextBuilder")


class ContextBuilder:
    """Build and cache IncidentContext objects from pipeline outputs.

    Parameters
    ----------
    explanations_path : str
        ``outputs/session_explanations.json`` (Phase 7).
    predictions_path : str
        ``outputs/attack_predictions.csv`` (Phase 6).
    dataset_path : str
        ``outputs/merged_dataset.csv`` (Phase 6).
    """

    def __init__(
        self,
        explanations_path: str = "outputs/session_explanations.json",
        predictions_path: str = "outputs/attack_predictions.csv",
        dataset_path: str = "outputs/merged_dataset.csv",
    ) -> None:
        self._exp_path  = Path(explanations_path)
        self._pred_path = Path(predictions_path)
        self._ds_path   = Path(dataset_path)

        # Internal caches
        self._exp_index:  Optional[Dict[str, dict]] = None   # session_id → explanation
        self._pred_index: Optional[Dict[str, dict]] = None   # session_id → prediction row
        self._ds_index:   Optional[Dict[str, dict]] = None   # session_id → dataset row
        self._ctx_cache:  Dict[str, IncidentContext] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "ContextBuilder":
        """Load and index all source files.  Returns self (fluent)."""
        t0 = time.time()
        self._load_explanations()
        self._load_predictions()
        self._load_dataset()
        logger.info("All source files loaded in %s.", elapsed_str(t0))
        return self

    def build(self, session_id: str) -> IncidentContext:
        """Build an IncidentContext for one session.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        IncidentContext
        """
        if session_id in self._ctx_cache:
            return self._ctx_cache[session_id]
        ctx = self._build_one(session_id)
        
        # Enforce RISK_ALERT_THRESHOLD checks
        from config.config import RISK_ALERT_THRESHOLD
        if ctx.risk_score < RISK_ALERT_THRESHOLD:
            ctx.attack_type = "Normal"
            ctx.severity = "Low"
            ctx.mitre = None
            ctx.investigation_steps = []
            ctx.nl_explanation = "No significant anomalies were detected. The session conforms to the user's established baseline."
            ctx.summary = "No significant anomalies were detected. The session conforms to the user's established baseline."
            if isinstance(ctx.copilot_context, dict):
                ctx.copilot_context["predicted_primary_attack"] = "Normal"
                ctx.copilot_context["severity"] = "Low"
                ctx.copilot_context["mitre"] = None
            ctx.top3_predictions = [
                {"attack": "Normal", "probability": round(ctx.confidence, 4)},
                {"attack": "Device Spoofing", "probability": 0.0},
                {"attack": "Lateral Movement", "probability": 0.0}
            ]

        self._ctx_cache[session_id] = ctx
        return ctx

    def build_all(self) -> List[IncidentContext]:
        """Build IncidentContext for every session in the explanations file."""
        self._ensure_loaded()
        t0 = time.time()
        contexts = []
        for sid in self._exp_index:
            contexts.append(self.build(sid))
        logger.info(
            "Built %d IncidentContext objects in %s.", len(contexts), elapsed_str(t0)
        )
        return contexts

    def get_anomalous(self) -> List[IncidentContext]:
        """Return only sessions classified as non-Normal."""
        return [c for c in self.build_all() if c.is_anomalous]

    def get_by_severity(self, severity: str) -> List[IncidentContext]:
        """Return sessions matching the given severity level."""
        return [c for c in self.build_all() if c.severity == severity]

    def get_high_critical(self) -> List[IncidentContext]:
        """Return High and Critical severity sessions."""
        return [c for c in self.build_all() if c.severity in ("High", "Critical")]

    @property
    def session_ids(self) -> List[str]:
        """All session IDs available in the explanations file."""
        self._ensure_loaded()
        return list(self._exp_index.keys())

    @property
    def total_sessions(self) -> int:
        self._ensure_loaded()
        return len(self._exp_index)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._exp_index is None:
            self.load()

    def _load_explanations(self) -> None:
        if not self._exp_path.exists():
            raise FileNotFoundError(f"Explanations not found: {self._exp_path}")
        logger.info("Loading %s …", self._exp_path)
        t0 = time.time()
        with open(self._exp_path, encoding="utf-8") as f:
            data = json.load(f)
        self._exp_index = {row["session_id"]: row for row in data}
        logger.info("  Loaded %d session explanations in %s.", len(self._exp_index), elapsed_str(t0))

    def _load_predictions(self) -> None:
        if not self._pred_path.exists():
            logger.warning("Predictions file not found: %s — top-3 will be empty.", self._pred_path)
            self._pred_index = {}
            return
        logger.info("Loading %s …", self._pred_path)
        df = pd.read_csv(self._pred_path)
        self._pred_index = {str(r["session_id"]): r.to_dict() for _, r in df.iterrows()}
        logger.info("  Loaded %d prediction rows.", len(self._pred_index))

    def _load_dataset(self) -> None:
        if not self._ds_path.exists():
            logger.warning("Dataset not found: %s — employee_id and temporal fields will be empty.", self._ds_path)
            self._ds_index = {}
            return
        logger.info("Loading %s (sampling metadata columns) …", self._ds_path)
        # Only load necessary metadata columns to save memory
        needed = [
            "session_id", "employee_id", "session_start_hour",
            "session_duration_minutes", "risk_score",
        ]
        df = pd.read_csv(self._ds_path, usecols=lambda c: c in needed)
        self._ds_index = {str(r["session_id"]): r.to_dict() for _, r in df.iterrows()}
        logger.info("  Loaded %d dataset rows.", len(self._ds_index))

    def _build_one(self, session_id: str) -> IncidentContext:
        self._ensure_loaded()

        exp = self._exp_index.get(session_id, {})
        pred = self._pred_index.get(session_id, {})
        ds   = self._ds_index.get(session_id, {})

        # Top-3 predictions from predictions CSV
        top3: list = []
        for i in (1, 2, 3):
            lbl = pred.get(f"top_{i}")
            prob = pred.get(f"top_{i}_prob")
            if lbl:
                top3.append({"attack": str(lbl), "probability": round(float(prob or 0), 4)})

        # Risk score — prefer dataset (full data) over explanation
        risk_score = float(ds.get("risk_score") or exp.get("risk_score") or 0.0)

        ctx = IncidentContext(
            session_id            = session_id,
            employee_id           = str(ds.get("employee_id") or "Unknown"),
            attack_type           = str(exp.get("prediction", "Normal")),
            confidence            = float(exp.get("confidence", 0.5)),
            severity              = str(exp.get("severity", "Low")),
            risk_score            = risk_score,
            anomaly_score         = float(exp.get("anomaly_score", 0.0)),
            mitre                 = exp.get("mitre"),
            positive_contributors = exp.get("positive_contributors", []),
            negative_contributors = exp.get("negative_contributors", []),
            investigation_steps   = exp.get("investigation_steps", []),
            nl_explanation        = str(exp.get("nl_explanation", "")),
            summary               = str(exp.get("summary", "")),
            copilot_context       = exp.get("copilot_context", {}),
            top3_predictions      = top3,
            session_start_hour    = _safe_int(ds.get("session_start_hour")),
            session_duration      = _safe_float(ds.get("session_duration_minutes")),
            true_label            = str(exp.get("true_label", "")),
        )
        return ctx


def _safe_int(val) -> Optional[int]:
    try:
        return int(float(val)) if val is not None and str(val) != "nan" else None
    except (ValueError, TypeError):
        return None


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None and str(val) != "nan" else None
    except (ValueError, TypeError):
        return None
