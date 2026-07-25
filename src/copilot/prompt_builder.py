"""
prompt_builder.py – Structured prompt templates for the AI Security Copilot.

Builds formatted prompts for the LLM, injecting IncidentContext and specifying
output schemas, instructions, and professional SOC analyst roles.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

from copilot.utils import IncidentContext

logger = logging.getLogger("Copilot.PromptBuilder")


class PromptBuilder:
    """Creates structured prompts from IncidentContext templates."""

    @staticmethod
    def build_system_prompt() -> str:
        return (
            "You are an expert Security Operations Centre (SOC) Tier-3 analyst and Incident Responder. "
            "Your task is to analyze ML-detected anomalies and SHAP explainability metrics to produce "
            "grounded, professional, and highly actionable incident reports, summaries, and containment playbooks. "
            "You must adhere strictly to the following rules:\n"
            "1. Grounding: Answer using ONLY the provided incident context and evidence. Do not invent details.\n"
            "2. Professionalism: Write in a precise, objective, and clear cybersecurity style.\n"
            "3. Format: Structure your output using Markdown as requested. Do not include meta-commentary."
        )

    @classmethod
    def build_prompt(cls, template_name: str, ctx: IncidentContext, **kwargs: Any) -> str:
        """Build a prompt by inserting IncidentContext into a template.

        Parameters
        ----------
        template_name : str
            One of: "incident_report", "executive_summary", "technical_analysis", "analyst_qa", "remediation_plan".
        ctx : IncidentContext
        kwargs : dict
            Additional arguments (e.g. "question" or "history" for analyst_qa).

        Returns
        -------
        str
        """
        method = getattr(cls, f"_template_{template_name}", None)
        if not method:
            raise ValueError(f"Unknown prompt template: {template_name}")
        return method(ctx, **kwargs)

    # ------------------------------------------------------------------
    # Templates
    # ------------------------------------------------------------------

    @staticmethod
    def _template_incident_report(ctx: IncidentContext, **kwargs: Any) -> str:
        context_str = _serialize_context(ctx)
        return f"""Analyze the following incident context and generate a complete SOC Incident Report.

{context_str}

Ensure your report includes the following structured sections:
1. Executive Summary: High-level overview of the incident.
2. Incident Overview: Core metadata, user, device, and timing.
3. Attack Classification & Prediction: XGBoost predicted attack class and top 3 probabilities.
4. Severity Assessment: Risk, anomaly, and SHAP-based severity.
5. MITRE ATT&CK Mapping: Tactic, technique ID, and technique name.
6. Behavioral Indicators: What deviations occurred.
7. Key SHAP Evidence: Top contributing features.
8. Exonerating/Counteracting Factors: What features reduced the risk.
9. Risk Assessment: Cumulative risk score evaluation.
10. Potential Business Impact: Impact description.
11. Containment Actions: Playbook actions for this attack type.
12. Investigation Checklist: Forensic tasks.
13. Remediation Plan: Long-term remediation.
14. Recovery Recommendations: Recovery actions.
15. Lessons Learned: Future preventative recommendations.

Output the report in clean Markdown."""

    @staticmethod
    def _template_executive_summary(ctx: IncidentContext, **kwargs: Any) -> str:
        context_str = _serialize_context(ctx)
        return f"""Review this incident context and write an Executive Summary for executive leadership.

{context_str}

Instructions:
- Keep the summary to 150 words or less.
- Avoid low-level technical jargon where possible.
- Clearly state the business impact and the primary containment action taken or recommended.
- Write in a professional, authoritative tone."""

    @staticmethod
    def _template_technical_analysis(ctx: IncidentContext, **kwargs: Any) -> str:
        context_str = _serialize_context(ctx)
        return f"""Perform a Technical Analysis of the following ML-detected incident for a senior SOC analyst.

{context_str}

Instructions:
- Deconstruct the feature values and SHAP impacts.
- Explain the behavioral significance of the top contributing features.
- Connect the anomalies to the mapped MITRE technique.
- Discuss how the confidence score relates to the predictions list."""

    @staticmethod
    def _template_analyst_qa(ctx: IncidentContext, **kwargs: Any) -> str:
        context_str = _serialize_context(ctx)
        question = kwargs.get("question", "No question provided.")
        history = kwargs.get("history", "")
        
        history_block = f"\nConversation History:\n{history}\n" if history else ""

        return f"""You are answering an analyst's question about the following incident context.

{context_str}
{history_block}
Analyst's Question:
"{question}"

Instructions:
- Answer the question directly and concisely.
- Reference specific metrics (e.g. SHAP values, risk scores, confidence) from the context.
- Ground all facts in the provided context. If the context does not contain the answer, say "Based on the available incident context, this information is not recorded." Do not make assumptions."""

    @staticmethod
    def _template_remediation_plan(ctx: IncidentContext, **kwargs: Any) -> str:
        context_str = _serialize_context(ctx)
        return f"""Develop a structured Remediation Plan for the following incident.

{context_str}

Organize the plan into 4 chronological phases:
1. Immediate Containment (Actions to take in minutes)
2. Forensic Investigation (Actions to take in hours)
3. Remediation (Actions to take in days)
4. Recovery & Lessons Learned (Actions to take in weeks)

Provide clear, bulleted steps for each phase."""


def _serialize_context(ctx: IncidentContext) -> str:
    """Formats IncidentContext fields into a readable text block for LLM prompt injection."""
    pos_features = "\n".join(
        [f"  - {c['feature']}: value={c['value']}, SHAP={c['shap_value']:.4f} ({c['impact']})"
         for c in ctx.positive_contributors[:8]]
    )
    neg_features = "\n".join(
        [f"  - {c['feature']}: value={c['value']}, SHAP={c['shap_value']:.4f} ({c['impact']})"
         for c in ctx.negative_contributors[:5]]
    )
    top3_str = ", ".join([f"{p['attack']} ({p['probability']:.0%})" for p in ctx.top3_predictions])
    steps_str = "\n".join([f"  - {step}" for step in ctx.investigation_steps])

    # Multi-behaviour context (injected by CombinedSimulator)
    detected_behaviours = (ctx.copilot_context or {}).get("detected_behaviours", [])
    behaviours_block = ""
    if detected_behaviours:
        behaviour_list = "\n".join([f"  • {b}" for b in detected_behaviours])
        behaviours_block = f"\nDetected Behaviours (multi-stage attack chain):\n{behaviour_list}\nPredicted Primary Attack: {ctx.attack_type} (Confidence: {ctx.confidence:.1%})\n"

    return f"""=== INCIDENT CONTEXT ===
Session ID: {ctx.session_id}
Employee ID: {ctx.employee_id}
XGBoost Prediction: {ctx.attack_type}
Confidence: {ctx.confidence:.1%}
Top 3 Class Predictions: {top3_str}
Severity: {ctx.severity}
Risk Score: {ctx.risk_score:.2f}
GRU Anomaly Score: {ctx.anomaly_score:.4f}
Source IP: {ctx.source_ip or 'N/A'}
Device ID: {ctx.device_id or 'N/A'}
Session Hour: {ctx.session_start_hour or 'N/A'}
Session Duration: {ctx.session_duration or 'N/A'} minutes
{behaviours_block}
MITRE ATT&CK Mapping:
  Tactic: {ctx.mitre_tactic or 'N/A'}
  Technique: {ctx.mitre_technique_id or 'N/A'} - {ctx.mitre_technique or 'N/A'}

Top Behavioral Anomalies (SHAP Positive Contributors):
{pos_features or '  (None)'}

Mitigating/Counteracting Behaviors (SHAP Negative Contributors):
{neg_features or '  (None)'}

Default Investigation Checklist:
{steps_str or '  (None)'}

Baseline NL Explanation:
{ctx.nl_explanation}
======================="""
