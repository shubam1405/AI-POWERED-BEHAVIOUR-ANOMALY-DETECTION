"""
report_generator.py – Generates full 15-section SOC incident reports.

Supports both Markdown and JSON-ready dictionary output.
Integrates template-based fallback when no LLM is configured.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from copilot.utils import IncidentContext, RecommendationSet
from copilot.llm_client import LLMClient
from copilot.prompt_builder import PromptBuilder
from copilot.recommendations import RecommendationEngine

logger = logging.getLogger("Copilot.ReportGenerator")


class ReportGenerator:
    """Orchestrates structured incident report generation."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.rec_engine = RecommendationEngine()

    def generate_report(self, ctx: IncidentContext) -> Dict[str, Any]:
        """Generate a complete 15-section SOC report as a dictionary.

        Parameters
        ----------
        ctx : IncidentContext

        Returns
        -------
        dict  Report sections map plus metadata.
        """
        # Generate structured playbooks first
        recs = self.rec_engine.generate(ctx)
        
        # Build prompt & system message
        prompt = PromptBuilder.build_prompt("incident_report", ctx)
        system = PromptBuilder.build_system_prompt()

        # Generate response using LLM or template fallback
        if self.llm.provider_name == "TemplateLLMClient":
            report_text = self._generate_template_report(ctx, recs)
        else:
            try:
                report_text = self.llm.generate(prompt=prompt, system=system)
            except Exception as e:
                logger.error("LLM report generation failed: %s. Falling back to template.", e)
                report_text = self._generate_template_report(ctx, recs)

        # Parse text into structured sections or wrap as whole
        sections = self._parse_report_sections(report_text)

        # Build clean JSON-serializable report object
        return {
            "session_id": ctx.session_id,
            "employee_id": ctx.employee_id,
            "attack_type": ctx.attack_type,
            "severity": ctx.severity,
            "confidence": ctx.confidence,
            "risk_score": ctx.risk_score,
            "anomaly_score": ctx.anomaly_score,
            "mitre": ctx.mitre,
            "report_text_markdown": report_text,
            "sections": sections,
            "recommendations": {
                "immediate_containment": recs.immediate_containment,
                "investigation": recs.investigation,
                "remediation": recs.remediation,
                "recovery": recs.recovery,
                "priority_action": recs.priority_action,
            },
        }

    def _generate_template_report(self, ctx: IncidentContext, recs: RecommendationSet) -> str:
        """Fully-populated high-quality report for zero-dependency template fallback."""
        pos_bullets = "\n".join(
            [f"- **{c['feature']}**: Value of {c['value']} introduced a {c['impact'].lower()} deviation."
             for c in ctx.positive_contributors[:6]]
        )
        neg_bullets = "\n".join(
            [f"- **{c['feature']}**: Value of {c['value']} acted as a mitigating factor ({c['impact'].lower()})."
             for c in ctx.negative_contributors[:4]]
        )
        
        mitre_str = (
            f"{ctx.mitre_tactic} ({ctx.mitre_technique_id} - {ctx.mitre_technique})"
            if ctx.mitre else "None mapped (Normal Activity)"
        )

        top3_predictions = ", ".join([f"{p['attack']} ({p['probability']:.0%})" for p in ctx.top3_predictions])

        return f"""# SOC Incident Report: {ctx.session_id}

## 1. Executive Summary
The Cyber Cage UEBA engine detected anomalous behaviors in session **{ctx.session_id}** associated with Employee **{ctx.employee_id}**. The session was classified as **{ctx.attack_type}** with a confidence score of **{ctx.confidence:.1%}**, indicating a **{ctx.severity}** severity risk. Immediate containment action is recommended.

## 2. Incident Overview
- **Session ID:** {ctx.session_id}
- **Employee ID:** {ctx.employee_id}
- **Source IP Address:** {ctx.source_ip or "Not available"}
- **Device ID:** {ctx.device_id or "Not available"}
- **Hour of Day:** {ctx.session_start_hour if ctx.session_start_hour is not None else "Unknown"}
- **Duration:** {f"{ctx.session_duration:.1f} minutes" if ctx.session_duration is not None else "Unknown"}

## 3. Attack Classification & Prediction
- **Primary Prediction:** {ctx.attack_type}
- **Classifier Confidence:** {ctx.confidence:.1%}
- **Top 3 Candidate Classes:** {top3_predictions or "N/A"}

## 4. Severity Assessment
The security risk level is determined to be **{ctx.severity}**. This is based on a cumulative risk score of **{ctx.risk_score:.2f}** and an isolated reconstruction reconstruction error of **{ctx.anomaly_score:.4f}** from the GRU autoencoder network.

## 5. MITRE ATT&CK Mapping
- **Tactic:** {ctx.mitre_tactic or "N/A"}
- **Technique:** {ctx.mitre_technique_id or "N/A"} - {ctx.mitre_technique or "N/A"}
- **Mapping Context:** Mapped directly based on {mitre_str}.

## 6. Behavioral Indicators
The session exhibited significant deviations from historical employee baselines. Key anomalous variables include:
- Total risk score: {ctx.risk_score}
- Anomaly score: {ctx.anomaly_score}
- Mapped attack signature similarity: {ctx.confidence:.1%}

## 7. Key SHAP Evidence
The following behavioral features contributed most significantly to the anomaly detection:
{pos_bullets or "- No positive contributors identified."}

## 8. Exonerating/Counteracting Factors
The following features displayed baseline behaviors that counteracted the threat classification:
{neg_bullets or "- No mitigating behaviors recorded."}

## 9. Risk Assessment
With a risk score of **{ctx.risk_score:.2f}**, this session represents a marked deviation from regular activity. The GRU network's anomaly index (**{ctx.anomaly_score:.4f}**) verifies that sequence reconstruction was highly corrupted, establishing objective evidence of behavioral drift.

## 10. Potential Business Impact
{recs.estimated_impact}

## 11. Containment Actions
The security operations team must perform these immediate containment steps:
{"".join([f"- [ ] {action}\\n" for action in recs.immediate_containment])}

## 12. Investigation Checklist
Forensic and triage investigative checklist:
{"".join([f"- [ ] {step}\\n" for step in recs.investigation])}

## 13. Remediation Plan
Follow-up remediation steps to resolve root causes:
{"".join([f"- [ ] {step}\\n" for step in recs.remediation])}

## 14. Recovery Recommendations
Systems recovery playbooks:
{"".join([f"- [ ] {step}\\n" for step in recs.recovery])}

## 15. Lessons Learned
- Verify user awareness concerning appropriate access policies.
- Audit resource permission trees regularly to enforce least-privilege policies.
- Deploy additional telemetry triggers based on the top SHAP indicators identified in this incident.
"""

    def _parse_report_sections(self, text: str) -> Dict[str, str]:
        """Tries to split the markdown report into discrete dictionary keys."""
        sections = {}
        # Simple parser looking for headers: e.g. "## 1. Executive Summary"
        import re
        pattern = r"##\s+(?:\d+\.)?\s*([^\n]+)\n(.*?)(?=\n##\s+(?:\d+\.)?\s*|\Z)"
        matches = re.findall(pattern, text, re.DOTALL)
        for title, content in matches:
            key = title.strip().lower().replace(" ", "_").replace("&", "and")
            sections[key] = content.strip()
        
        # Fallback if regex split fails
        if not sections:
            sections["full_text"] = text
        return sections
