"""
recommendations.py – Enforces 4-tier security playbooks based on predicted attack, severity, and SHAP.

Containment playbooks are pulled from copilot.utils and augmented dynamically based on
severity level (e.g. Critical severity enforces endpoint isolation and executive notification).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from copilot.utils import IncidentContext, CONTAINMENT_PLAYBOOKS, RecommendationSet

logger = logging.getLogger("Copilot.RecommendationEngine")


class RecommendationEngine:
    """Generates playbooks and immediate actions for detected incidents."""

    def __init__(self) -> None:
        pass

    def generate(self, ctx: IncidentContext) -> RecommendationSet:
        """Create a tailored playbook using the context metadata and SHAP features.

        Parameters
        ----------
        ctx : IncidentContext

        Returns
        -------
        RecommendationSet
        """
        attack = ctx.attack_type
        severity = ctx.severity

        # Fallback to general playbooks if specific attack type not registered
        playbook = CONTAINMENT_PLAYBOOKS.get(attack) or CONTAINMENT_PLAYBOOKS.get("Normal")
        
        # Make a copy to avoid mutating global dictionary
        imm = list(playbook.get("immediate_containment", []))
        inv = list(playbook.get("investigation", []))
        rem = list(playbook.get("remediation", []))
        rec = list(playbook.get("recovery", []))

        # Dynamic severity adjustments
        if severity == "Critical":
            # Escalate actions
            if "Isolate the affected endpoint from the network immediately" not in imm and attack != "Normal":
                imm.insert(0, "Isolate the affected endpoint from the network immediately")
            imm.append("Trigger emergency paging for CISO and senior leadership")
            inv.append("Initiate 24/7 continuous threat hunting across peer hosts")
        elif severity == "High":
            imm.append("Notify IT Security Manager immediately")
            inv.append("Perform full directory access audit for employee account")

        # Prioritize single action based on top contributing SHAP features
        priority = self._determine_priority_action(ctx, imm)
        
        # Business impact statement
        from copilot.utils import BUSINESS_IMPACT
        impact = BUSINESS_IMPACT.get(attack, "No business impact predicted.")

        return RecommendationSet(
            immediate_containment=imm,
            investigation=inv,
            remediation=rem,
            recovery=rec,
            priority_action=priority,
            estimated_impact=impact,
        )

    def _determine_priority_action(self, ctx: IncidentContext, immediate_actions: List[str]) -> str:
        """Select or construct the single most critical containment step."""
        if ctx.attack_type == "Normal":
            return "No containment actions required. Monitor baseline activity."

        # If there are positive contributors, customize priority action
        top_pos = ctx.positive_contributors[:3]
        features = [c["feature"] for c in top_pos]

        if "powershell_executions" in features or "powershell_position" in features:
            return "Terminate any active PowerShell or script execution processes on this host."
        if "total_upload_bytes" in features or "files_downloaded" in features:
            return "Block outbound network transfers and isolate the device immediately to prevent exfiltration."
        if "failed_login_count" in features:
            return "Revoke user credentials and disable active login sessions immediately."
        if "admin_resource_accesses" in features or "admin_access_position" in features:
            return "Temporarily revoke administrator rights and active admin sessions for this user."

        # Default fallback to first immediate containment action
        return immediate_actions[0] if immediate_actions else "Isolate and inspect the endpoint."
