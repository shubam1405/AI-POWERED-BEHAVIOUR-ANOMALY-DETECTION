"""
incident_summarizer.py – Generates targeted summaries for multiple audiences.

Supports:
    - Executive Summary (max 150 words)
    - SOC Summary (technical details + evidence)
    - Dashboard Summary (2-3 sentences max)
    - Email Alert Summary (subject + 5 bullets)
"""

from __future__ import annotations

import logging
from typing import Dict

from copilot.utils import IncidentContext
from copilot.llm_client import LLMClient
from copilot.prompt_builder import PromptBuilder

logger = logging.getLogger("Copilot.IncidentSummarizer")


class IncidentSummarizer:
    """Generates multi-audience incident summaries."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    def generate_all(self, ctx: IncidentContext) -> Dict[str, str]:
        """Generate all four summary formats for a single incident context.

        Parameters
        ----------
        ctx : IncidentContext

        Returns
        -------
        dict  Map containing executive, soc, dashboard, and email summaries.
        """
        # 1. Dashboard summary (always template-based for consistent length/format)
        dashboard = self._generate_dashboard_summary(ctx)

        # 2. Executive summary
        exec_prompt = PromptBuilder.build_prompt("executive_summary", ctx)
        if self.llm.provider_name == "TemplateLLMClient":
            exec_sum = self._generate_template_exec_summary(ctx)
        else:
            try:
                exec_sum = self.llm.generate(exec_prompt, PromptBuilder.build_system_prompt())
            except Exception as e:
                logger.error("LLM exec summary failed: %s. Using template.", e)
                exec_sum = self._generate_template_exec_summary(ctx)

        # 3. SOC summary
        soc_prompt = PromptBuilder.build_prompt("technical_analysis", ctx)
        if self.llm.provider_name == "TemplateLLMClient":
            soc_sum = self._generate_template_soc_summary(ctx)
        else:
            try:
                soc_sum = self.llm.generate(soc_prompt, PromptBuilder.build_system_prompt())
            except Exception as e:
                logger.error("LLM SOC summary failed: %s. Using template.", e)
                soc_sum = self._generate_template_soc_summary(ctx)

        # 4. Email Alert summary
        email_sum = self._generate_email_summary(ctx)

        return {
            "executive": exec_sum.strip(),
            "soc": soc_sum.strip(),
            "dashboard": dashboard.strip(),
            "email": email_sum.strip(),
        }

    # ------------------------------------------------------------------
    # Template-based fallback generators
    # ------------------------------------------------------------------

    def _generate_dashboard_summary(self, ctx: IncidentContext) -> str:
        """Dashboard summary: 2-3 sentences max."""
        if ctx.attack_type == "Normal":
            return f"Session {ctx.session_id} exhibits baseline behaviors with no threat indicators."
        
        top_feats = ctx.top_positive_features[:2]
        feats_str = " and ".join([f"'{f}'" for f in top_feats]) if top_feats else "behavioral markers"
        
        return (
            f"Anomalous session detected for employee {ctx.employee_id} classified as {ctx.attack_type} "
            f"({ctx.severity} severity, {ctx.confidence:.0%} confidence). "
            f"Key contributing indicators include elevated values for {feats_str}. "
            f"Immediate containment action: {ctx.priority_action}."
        )

    def _generate_template_exec_summary(self, ctx: IncidentContext) -> str:
        """Executive summary <= 150 words."""
        if ctx.attack_type == "Normal":
            return "No anomalous behavior detected. The session aligns with standard employee baseline activity."
            
        return (
            f"Executive Alert: A security anomaly was detected in session {ctx.session_id} (Employee {ctx.employee_id}). "
            f"The session exhibits behavioral patterns indicative of a {ctx.attack_type} attempt, carrying a "
            f"risk severity of {ctx.severity} and an ML classifier confidence of {ctx.confidence:.1%}. "
            f"The primary driver of this alert was anomaly detection scoring by the GRU autoencoder model "
            f"(score: {ctx.anomaly_score:.4f}) and feature deviations quantified by SHAP. "
            f"Potential business impact includes potential data compromise or unauthorized resource access. "
            f"The Security Operations team recommends executing immediate containment actions, specifically: "
            f"{ctx.priority_action}."
        )

    def _generate_template_soc_summary(self, ctx: IncidentContext) -> str:
        """Detailed technical summary for SOC analysts."""
        if ctx.attack_type == "Normal":
            return "Triage: Baseline session metrics. Normal activity."

        pos_lines = [
            f"  - {c['feature']} (value: {c['value']}, SHAP: {c['shap_value']:.4f}) -> {c['impact']}"
            for c in ctx.positive_contributors[:5]
        ]
        pos_str = "\n".join(pos_lines)

        mitre_str = (
            f"MITRE Tactic: {ctx.mitre_tactic} | Technique: {ctx.mitre_technique_id} ({ctx.mitre_technique})"
            if ctx.mitre else "No direct MITRE mapping."
        )

        return f"""[SOC TRIAGE SUMMARY]
Incident ID   : {ctx.session_id}
Target Subject : Employee {ctx.employee_id}
Prediction    : {ctx.attack_type} (Confidence: {ctx.confidence:.1%})
Risk Score    : {ctx.risk_score:.2f} (GRU Anomaly Score: {ctx.anomaly_score:.4f})
Severity      : {ctx.severity}
{mitre_str}

Behavioral Evidence (SHAP contributors):
{pos_str}

Summary & Action:
The user session deviated significantly from the employee's standard profile, specifically showing anomalous feature shifts. 
This aligns with the classifier's signature for {ctx.attack_type}.
Priority Containment Step: {ctx.priority_action}"""

    def _generate_email_summary(self, ctx: IncidentContext) -> str:
        """Subject + 5 bullets for security notifications."""
        if ctx.attack_type == "Normal":
            return "Subject: [INFO] Baseline Activity Monitored - Session " + ctx.session_id
            
        mitre_tag = f" [{ctx.mitre_technique_id}]" if ctx.mitre else ""
        subject = f"Subject: [ALERT - {ctx.severity.upper()}] {ctx.attack_type}{mitre_tag} Detected (Session: {ctx.session_id})"
        
        top_feats = [c["feature"] for c in ctx.positive_contributors[:3]]
        feats_str = ", ".join(top_feats) if top_feats else "None"

        return f"""{subject}

Security Operations Alert:
• Incident ID: {ctx.session_id} (Employee: {ctx.employee_id})
• Classification: {ctx.attack_type} (Confidence: {ctx.confidence:.0%}) | Severity: {ctx.severity}
• Detection Metrics: Risk Score = {ctx.risk_score:.2f} | Anomaly Score = {ctx.anomaly_score:.4f}
• Core Indicators: {feats_str}
• Recommended Containment: {ctx.priority_action}"""
