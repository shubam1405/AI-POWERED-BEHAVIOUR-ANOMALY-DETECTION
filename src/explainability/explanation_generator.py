"""
explanation_generator.py – Natural language explanation builder.

Converts a structured explanation dict (from ``LocalExplainer``) into:

1. ``nl_explanation`` – multi-sentence paragraph for SOC analysts.
2. ``summary``        – ≤ 2 sentences for dashboard cards and copilot context.

No LLM is required.  All text is generated deterministically from
templates + the knowledge base in ``utils.py``.  SHAP terminology is
never surfaced in the final output.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from explainability.utils import ATTACK_NL_CONTEXT, get_feature_description

logger = logging.getLogger("Explainability.ExplanationGenerator")

# Maximum positive contributors to mention in the explanation body
_MAX_BULLETS = 6

# Severity → introductory tone adjective
_SEVERITY_TONE: Dict[str, str] = {
    "Critical": "extremely high",
    "High":     "high",
    "Medium":   "moderate",
    "Low":      "low",
}

# Direction qualifiers for feature bullets
_DIRECTION_PHRASES: Dict[str, str] = {
    "High Positive":   "extremely high",
    "Medium Positive": "elevated",
    "Low Positive":    "slightly elevated",
    "Negligible":      "unchanged",
    "Low Negative":    "slightly reduced",
    "Medium Negative": "below normal",
    "High Negative":   "significantly below normal",
}


class ExplanationGenerator:
    """Generate natural-language explanations from structured explanation dicts.

    Parameters
    ----------
    attack_context : dict, optional
        Override for ``ATTACK_NL_CONTEXT`` (useful for testing).
    """

    def __init__(
        self,
        attack_context: Dict[str, str] | None = None,
    ) -> None:
        self._context = attack_context or ATTACK_NL_CONTEXT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Add ``nl_explanation`` and ``summary`` fields to *explanation* in-place.

        Parameters
        ----------
        explanation : dict
            Output from ``LocalExplainer.explain_session()``.

        Returns
        -------
        dict  (same dict, mutated)
        """
        prediction   = explanation.get("prediction",   "Normal")
        confidence   = float(explanation.get("confidence",   0.5))
        severity     = explanation.get("severity",     "Low")
        anomaly_sc   = float(explanation.get("anomaly_score", 0.0))
        pos_contribs = explanation.get("positive_contributors", [])
        neg_contribs = explanation.get("negative_contributors", [])
        mitre        = explanation.get("mitre")

        explanation["nl_explanation"] = self._build_nl(
            prediction, confidence, severity, anomaly_sc,
            pos_contribs, neg_contribs, mitre,
        )
        explanation["summary"] = self._build_summary(
            prediction, confidence, severity, pos_contribs,
        )
        return explanation

    def generate_batch(self, explanations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate NL for a list of explanation dicts."""
        for exp in explanations:
            self.generate(exp)
        logger.info("NL explanations generated for %d sessions.", len(explanations))
        return explanations

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    def _build_nl(
        self,
        prediction: str,
        confidence: float,
        severity: str,
        anomaly_score: float,
        pos_contribs: List[Dict],
        neg_contribs: List[Dict],
        mitre: Dict | None,
    ) -> str:
        lines: List[str] = []

        # --- Header ---
        tone = _SEVERITY_TONE.get(severity, "moderate")
        conf_pct = f"{confidence:.0%}"
        lines.append(
            f"The model classified this session as **{prediction}** with "
            f"{conf_pct} confidence ({severity} severity)."
        )

        # --- Attack context ---
        ctx = self._context.get(prediction, "")
        if ctx:
            lines.append(ctx)

        # --- MITRE reference ---
        if mitre:
            lines.append(
                f"This maps to MITRE ATT&CK tactic **{mitre['tactic']}** "
                f"({mitre['technique_id']} – {mitre['technique']})."
            )

        # --- Anomaly score ---
        if anomaly_score > 0.01:
            lines.append(
                f"The session's behavioural anomaly score is "
                f"{anomaly_score:.2f}, indicating a {tone} deviation "
                f"from the user's established baseline."
            )

        # --- Positive contributors ---
        bullets = self._build_bullets(pos_contribs[:_MAX_BULLETS], positive=True)
        if bullets:
            lines.append("Key indicators that increased the prediction:")
            lines.extend(bullets)

        # --- Negative contributors ---
        neg_bullets = self._build_bullets(neg_contribs[:3], positive=False)
        if neg_bullets:
            lines.append("Factors that partially counteracted the prediction:")
            lines.extend(neg_bullets)

        # --- Closing ---
        if prediction != "Normal":
            lines.append(
                f"These indicators collectively resemble historical "
                f"**{prediction}** attack patterns observed in the training data."
            )

        return "\n\n".join(lines)

    def _build_bullets(
        self,
        contributors: List[Dict],
        positive: bool,
    ) -> List[str]:
        bullets: List[str] = []
        for c in contributors:
            feature     = c.get("feature", "")
            description = c.get("description", get_feature_description(feature))
            impact      = c.get("impact", "")
            value       = c.get("value")
            direction   = _DIRECTION_PHRASES.get(impact, "")
            category    = c.get("category", "")

            # Build natural phrase
            val_str = ""
            if value is not None:
                try:
                    fv = float(value)
                    if fv == int(fv):
                        val_str = f" (value: {int(fv)})"
                    else:
                        val_str = f" (value: {fv:.2f})"
                except (ValueError, TypeError):
                    pass

            if direction:
                bullet = f"• {description.capitalize()}{val_str} was {direction} [{category}]"
            else:
                arrow = "↑" if positive else "↓"
                bullet = f"• {description.capitalize()}{val_str} {arrow} [{category}]"

            bullets.append(bullet)
        return bullets

    def _build_summary(
        self,
        prediction: str,
        confidence: float,
        severity: str,
        pos_contribs: List[Dict],
    ) -> str:
        """Build a ≤ 2-sentence copilot-ready summary."""
        conf_pct = f"{confidence:.0%}"

        if prediction == "Normal":
            return (
                "No significant anomalies were detected. "
                "The session conforms to the user's established baseline."
            )

        # Top 2–3 feature descriptions
        top_feats = [
            c.get("description", c.get("feature", ""))
            for c in pos_contribs[:3]
        ]

        if top_feats:
            feat_str = ", ".join(top_feats[:2])
            if len(top_feats) >= 3:
                feat_str += f", and {top_feats[2]}"
            sent1 = (
                f"{feat_str.capitalize()} strongly indicate "
                f"**{prediction}** ({severity} severity, {conf_pct} confidence)."
            )
        else:
            sent1 = (
                f"Behavioural anomalies strongly indicate "
                f"**{prediction}** ({severity} severity, {conf_pct} confidence)."
            )

        ctx = self._context.get(prediction, "")
        sent2 = ctx.split(".")[0] + "." if ctx else ""

        return f"{sent1} {sent2}".strip()
