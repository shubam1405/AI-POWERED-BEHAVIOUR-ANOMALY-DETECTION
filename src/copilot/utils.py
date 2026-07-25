"""
utils.py – Shared knowledge base, dataclasses, and helpers for the AI Security Copilot.

Contains:
    - IncidentContext dataclass (unified incident object)
    - BUSINESS_IMPACT descriptions per attack type
    - CONTAINMENT_PLAYBOOKS (4-tier) per attack type
    - ATTACK_CHAIN_SEQUENCES for correlation
    - Common helpers: ensure_dir, elapsed_str, sanitize_filename
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("Copilot.Utils")


# ---------------------------------------------------------------------------
# IncidentContext — unified incident object
# ---------------------------------------------------------------------------

@dataclass
class IncidentContext:
    """Single source of truth for one session's incident data."""

    session_id:             str
    employee_id:            str = "Unknown"
    attack_type:            str = "Normal"
    confidence:             float = 0.5
    severity:               str = "Low"
    risk_score:             float = 0.0
    anomaly_score:          float = 0.0
    mitre:                  Optional[Dict[str, str]] = None
    positive_contributors:  List[Dict[str, Any]] = field(default_factory=list)
    negative_contributors:  List[Dict[str, Any]] = field(default_factory=list)
    investigation_steps:    List[str] = field(default_factory=list)
    nl_explanation:         str = ""
    summary:                str = ""
    copilot_context:        Dict[str, Any] = field(default_factory=dict)
    top3_predictions:       List[Dict[str, Any]] = field(default_factory=list)
    session_start_hour:     Optional[int] = None
    session_duration:       Optional[float] = None
    source_ip:              Optional[str] = None
    device_id:              Optional[str] = None
    timestamp:              Optional[str] = None
    true_label:             str = ""

    @property
    def is_anomalous(self) -> bool:
        return self.attack_type != "Normal"

    @property
    def mitre_tactic(self) -> str:
        return (self.mitre or {}).get("tactic", "")

    @property
    def mitre_technique_id(self) -> str:
        return (self.mitre or {}).get("technique_id", "")

    @property
    def mitre_technique(self) -> str:
        return (self.mitre or {}).get("technique", "")

    @property
    def top_positive_features(self) -> List[str]:
        return [c["feature"] for c in self.positive_contributors[:5]]

    @property
    def priority_action(self) -> str:
        steps = self.investigation_steps
        return steps[0] if steps else "Investigate the session activity."


@dataclass
class RecommendationSet:
    """Chronologically-ordered response actions for security triage and recovery."""
    immediate_containment: List[str]
    investigation:         List[str]
    remediation:           List[str]
    recovery:              List[str]
    priority_action:       str
    estimated_impact:      str


# ---------------------------------------------------------------------------
# Business Impact Descriptions
# ---------------------------------------------------------------------------

BUSINESS_IMPACT: Dict[str, str] = {
    "Normal": "No business impact detected.",
    "Beaconing C2": (
        "Active C2 communication indicates a host is compromised and under attacker control. "
        "Attackers may exfiltrate data, deploy ransomware, or use the host as a pivot point. "
        "Potential impact: intellectual property theft, regulatory breach, reputational damage."
    ),
    "Brute Force": (
        "Credential compromise may allow unauthorised access to systems and sensitive data. "
        "If successful, attackers gain persistent access. "
        "Potential impact: data breach, compliance violation, account takeover."
    ),
    "Credential Stuffing": (
        "Automated credential attacks using breached credentials. "
        "Successful logins could expose customer or employee data. "
        "Potential impact: account takeover, data breach, regulatory fines."
    ),
    "Data Exfiltration": (
        "Sensitive data is actively leaving the organisation. "
        "This may constitute a notifiable breach under GDPR, HIPAA, or equivalent regulations. "
        "Potential impact: regulatory fines, competitive disadvantage, customer trust erosion."
    ),
    "Device Spoofing": (
        "Attacker is masquerading as a trusted device to bypass security controls. "
        "Potential impact: unauthorised access, security control bypass, data theft."
    ),
    "Impossible Travel": (
        "Account credentials may have been stolen and used from a remote location. "
        "Potential impact: full account compromise, data breach, business email compromise."
    ),
    "Insider Drift": (
        "Gradual behaviour change may indicate growing insider threat. "
        "Potential impact: data leakage, sabotage, compliance violation."
    ),
    "Insider Threat": (
        "Malicious insider activity could expose sensitive intellectual property or customer data. "
        "Potential impact: data theft, sabotage, reputational and financial damage."
    ),
    "Lateral Movement": (
        "Attacker is expanding their foothold across the network. "
        "Early containment is critical to prevent full network compromise. "
        "Potential impact: ransomware deployment, mass data theft, prolonged outage."
    ),
    "Low-and-Slow Exfiltration": (
        "Data is being quietly siphoned over an extended period to evade detection. "
        "Significant data loss may have already occurred. "
        "Potential impact: long-term IP theft, undiscovered breach, regulatory liability."
    ),
    "Malware Execution": (
        "Active malware on an endpoint poses immediate risk of propagation. "
        "Potential impact: ransomware, data destruction, network-wide compromise."
    ),
    "Off-hours Access": (
        "Unusual access outside business hours may indicate compromised credentials or an insider threat. "
        "Potential impact: unauthorised data access, policy violation, compliance breach."
    ),
    "Privilege Escalation": (
        "Attacker has gained or is attempting to gain elevated privileges. "
        "Admin-level access enables full system control. "
        "Potential impact: complete system compromise, mass data exfiltration."
    ),
    "Suspicious PowerShell": (
        "PowerShell abuse is a primary technique in ransomware and APT attacks. "
        "Scripts may be establishing persistence or downloading payloads. "
        "Potential impact: ransomware, credential theft, persistent backdoor."
    ),
    "USB Data Theft": (
        "Physical data theft via USB bypasses network-based controls entirely. "
        "Potential impact: intellectual property loss, compliance breach, irreversible data leak."
    ),
}


# ---------------------------------------------------------------------------
# Containment Playbooks (4-tier)
# ---------------------------------------------------------------------------

CONTAINMENT_PLAYBOOKS: Dict[str, Dict[str, List[str]]] = {
    "Beaconing C2": {
        "immediate_containment": [
            "Isolate the affected endpoint from the network immediately",
            "Block all outbound connections to identified C2 domains and IPs",
            "Terminate any suspicious processes identified on the endpoint",
            "Preserve a memory dump and disk image for forensic analysis",
        ],
        "investigation": [
            "Analyse network traffic captures for C2 communication patterns",
            "Review DNS query logs for newly registered or suspicious domains",
            "Identify the malware family using sandbox detonation",
            "Determine initial infection vector (phishing, exploit, supply chain)",
            "Check for lateral movement indicators from the compromised host",
        ],
        "remediation": [
            "Remove malware and persistence mechanisms from the endpoint",
            "Reset all credentials that were accessible from the compromised host",
            "Update endpoint protection signatures and detection rules",
            "Block identified IOCs across all network controls",
            "Patch the vulnerability used for initial compromise if identified",
        ],
        "recovery": [
            "Restore endpoint from a clean backup taken before compromise",
            "Re-enrol device with enhanced monitoring enabled",
            "Monitor for re-infection over the next 30 days",
            "Conduct post-incident review and update SOC playbooks",
        ],
    },
    "Brute Force": {
        "immediate_containment": [
            "Lock the targeted account(s) immediately",
            "Block the source IP address at the perimeter firewall",
            "Enable MFA on the targeted account if not already enforced",
            "Alert the account owner via a secondary channel",
        ],
        "investigation": [
            "Determine whether any login attempts succeeded",
            "Review all activity from the source IP across all accounts",
            "Check the source IP against threat intelligence feeds",
            "Identify whether multiple accounts were targeted simultaneously",
        ],
        "remediation": [
            "Enforce account lockout policy after N failed attempts",
            "Implement CAPTCHA or rate-limiting on the authentication endpoint",
            "Enable MFA organisation-wide for all privileged accounts",
            "Reset the targeted account's password",
        ],
        "recovery": [
            "Unlock account after credentials are confirmed reset",
            "Monitor account for 7 days for follow-on anomalous activity",
            "Review and update authentication policy",
        ],
    },
    "Credential Stuffing": {
        "immediate_containment": [
            "Force password reset for all accounts targeted in the attack",
            "Block source IP ranges associated with the attack",
            "Implement temporary CAPTCHA on login endpoints",
            "Alert users whose accounts were targeted",
        ],
        "investigation": [
            "Cross-reference targeted usernames against known breach databases",
            "Determine whether any credential pairs successfully authenticated",
            "Review session activity for any successfully authenticated accounts",
            "Identify the credential source (purchased breach data, phishing kit)",
        ],
        "remediation": [
            "Deploy credential breach monitoring (e.g., HaveIBeenPwned integration)",
            "Enforce MFA for all user accounts",
            "Implement adaptive authentication based on risk scoring",
            "Block known credential stuffing proxy IP ranges",
        ],
        "recovery": [
            "Notify affected users per data breach notification requirements",
            "Monitor affected accounts for 30 days",
            "Review and harden authentication infrastructure",
        ],
    },
    "Data Exfiltration": {
        "immediate_containment": [
            "Disable the user account or revoke its network access immediately",
            "Block all outbound transfers to identified external destinations",
            "Preserve all access logs, network captures, and file access records",
            "Notify Legal and Compliance teams — this may be a notifiable breach",
        ],
        "investigation": [
            "Identify all files and data accessed and transferred in the session",
            "Determine the destination of the transferred data",
            "Review email, cloud storage, and USB activity in the same timeframe",
            "Interview the employee's manager about recent behaviour or grievances",
            "Determine whether the data constitutes personally identifiable information (PII)",
        ],
        "remediation": [
            "Implement DLP controls on email, cloud storage, and USB channels",
            "Apply need-to-know access controls to sensitive data repositories",
            "Revoke access to sensitive data pending investigation outcome",
            "Update data classification and handling policies",
        ],
        "recovery": [
            "Assess regulatory notification obligations (GDPR, HIPAA, etc.)",
            "Engage legal counsel and prepare breach notification if required",
            "Conduct data impact assessment",
            "Implement enhanced monitoring for the employee and peer group",
        ],
    },
    "Device Spoofing": {
        "immediate_containment": [
            "Revoke trust for the spoofed device certificate or fingerprint",
            "Force re-authentication from a verified device",
            "Block sessions originating from unrecognised device fingerprints",
        ],
        "investigation": [
            "Identify which device was being spoofed and how",
            "Review all sessions using the spoofed device identity",
            "Check for other accounts affected by the same spoofed device",
            "Determine whether the real device is also compromised",
        ],
        "remediation": [
            "Implement device certificate pinning",
            "Enhance device identity verification in the authentication flow",
            "Deploy endpoint attestation for privileged access",
        ],
        "recovery": [
            "Re-enrol affected device with a new certificate",
            "Monitor device identity events for 14 days",
        ],
    },
    "Impossible Travel": {
        "immediate_containment": [
            "Suspend the account pending investigation",
            "Terminate all active sessions for the account",
            "Require re-verification of identity via secondary channel",
            "Block login from the anomalous geographic location",
        ],
        "investigation": [
            "Contact the user directly to verify whether both login locations are legitimate",
            "Check whether a VPN or proxy explains the geographic discrepancy",
            "Review what resources were accessed from both locations",
            "Determine whether this is account sharing or credential theft",
        ],
        "remediation": [
            "Enforce geographic login policies and geo-fencing",
            "Enable conditional access policies based on location risk",
            "Require step-up authentication for logins from new locations",
        ],
        "recovery": [
            "Re-enable account after identity is verified",
            "Implement enhanced monitoring for the account for 30 days",
        ],
    },
    "Insider Drift": {
        "immediate_containment": [
            "Flag the account for enhanced monitoring",
            "Notify the employee's manager confidentially",
            "Review and potentially restrict access to the most sensitive resources",
        ],
        "investigation": [
            "Review behaviour trends over the past 30–90 days",
            "Determine whether behaviour aligns with a role change or personal circumstance",
            "Interview the manager about any recent performance or attitude changes",
            "Check for correlation with any high-risk data access events",
        ],
        "remediation": [
            "Implement just-in-time access for sensitive resources",
            "Conduct an access review for the employee's account",
            "Engage HR if behaviour indicates disengagement or grievance",
        ],
        "recovery": [
            "Continue enhanced monitoring for 90 days",
            "Conduct quarterly access reviews",
        ],
    },
    "Insider Threat": {
        "immediate_containment": [
            "Suspend the account immediately pending investigation",
            "Preserve all logs, emails, and file access records",
            "Notify HR and Legal immediately",
            "Revoke all system access including VPN and cloud services",
        ],
        "investigation": [
            "Conduct a full forensic review of the employee's devices and accounts",
            "Identify all data accessed, copied, or transmitted",
            "Review communications for evidence of intent or external coordination",
            "Engage Legal and HR for formal investigation process",
        ],
        "remediation": [
            "Follow HR and Legal process for formal disciplinary action",
            "Implement departing-employee procedures retroactively",
            "Review and tighten need-to-know access policies",
            "Implement data staging detection rules",
        ],
        "recovery": [
            "Assess and remediate any data that was exfiltrated",
            "Conduct a lessons-learned review of access controls",
            "Update insider threat programme",
        ],
    },
    "Lateral Movement": {
        "immediate_containment": [
            "Isolate all affected systems from the network",
            "Revoke credentials used for lateral movement",
            "Block internal SMB, RDP, and SSH traffic from the source host",
            "Activate incident response team immediately",
        ],
        "investigation": [
            "Map the full scope of lateral movement (all systems reached)",
            "Identify the original entry point and initial compromise",
            "Determine which credentials were used or stolen",
            "Check for persistence mechanisms on all affected systems",
        ],
        "remediation": [
            "Reset all potentially compromised credentials",
            "Remove attacker persistence from all affected systems",
            "Segment the network to limit future lateral movement",
            "Apply principle of least privilege to service accounts",
        ],
        "recovery": [
            "Restore affected systems from clean backups",
            "Conduct full network sweep for residual compromise",
            "Implement network segmentation improvements",
            "Run a tabletop exercise to validate incident response capability",
        ],
    },
    "Low-and-Slow Exfiltration": {
        "immediate_containment": [
            "Suspend the account and terminate active sessions",
            "Block all identified exfiltration channels",
            "Preserve all available logs for the full suspected exfiltration period",
        ],
        "investigation": [
            "Calculate the total volume of data exfiltrated over the full timeframe",
            "Identify the complete list of files and data accessed",
            "Determine the exfiltration destination and method",
            "Assess whether data constitutes a notifiable breach",
        ],
        "remediation": [
            "Implement DLP with cumulative volume thresholds",
            "Deploy user and entity behaviour analytics (UEBA) with longer lookback windows",
            "Apply data egress controls",
        ],
        "recovery": [
            "Engage legal counsel for breach notification assessment",
            "Conduct data classification review",
            "Implement long-term behavioural monitoring",
        ],
    },
    "Malware Execution": {
        "immediate_containment": [
            "Isolate the infected endpoint immediately",
            "Kill identified malicious processes",
            "Block identified malware IOCs at network controls",
            "Preserve memory dump for forensic analysis",
        ],
        "investigation": [
            "Submit samples to sandbox for dynamic analysis",
            "Identify the malware family and capabilities",
            "Determine initial infection vector",
            "Check for lateral movement or C2 activity from the endpoint",
            "Review all processes spawned by the malware",
        ],
        "remediation": [
            "Re-image the affected endpoint",
            "Update EDR and AV signatures",
            "Patch the exploited vulnerability",
            "Deploy application whitelisting on affected systems",
        ],
        "recovery": [
            "Restore from a clean pre-infection backup",
            "Re-enrol endpoint with enhanced monitoring",
            "Conduct threat hunting across the environment for similar IOCs",
        ],
    },
    "Off-hours Access": {
        "immediate_containment": [
            "Alert the account owner's manager immediately",
            "Verify with the employee via a secondary channel whether the access is legitimate",
            "If unconfirmed, suspend the session",
        ],
        "investigation": [
            "Review the resources accessed during the off-hours session",
            "Check the login location and device against known good values",
            "Determine whether any sensitive data was accessed",
        ],
        "remediation": [
            "Implement time-based access policies for sensitive resources",
            "Require step-up authentication for off-hours access",
        ],
        "recovery": [
            "Re-enable access once legitimacy is confirmed",
            "Monitor for recurrence",
        ],
    },
    "Privilege Escalation": {
        "immediate_containment": [
            "Revoke the escalated privileges immediately",
            "Suspend the account if unauthorised escalation is confirmed",
            "Remove any backdoors or persistence mechanisms added with elevated privileges",
        ],
        "investigation": [
            "Determine how privilege escalation was achieved (exploit, misconfiguration, credential theft)",
            "Identify all actions taken with the elevated privileges",
            "Check for other accounts affected by the same escalation vector",
        ],
        "remediation": [
            "Patch the vulnerability or misconfiguration used for escalation",
            "Implement just-in-time privileged access management (PAM)",
            "Enable privileged access workstations (PAW) for admin tasks",
            "Audit all privileged group memberships",
        ],
        "recovery": [
            "Review and remediate all changes made with elevated access",
            "Conduct a full privilege access review",
            "Implement privileged activity monitoring",
        ],
    },
    "Suspicious PowerShell": {
        "immediate_containment": [
            "Terminate all suspicious PowerShell processes",
            "Isolate the endpoint if encoded or download-cradle commands were detected",
            "Block outbound connections initiated by PowerShell",
        ],
        "investigation": [
            "Review Script Block Logging and Module Logging outputs",
            "Decode any Base64-encoded commands for analysis",
            "Identify whether any payloads were downloaded or executed",
            "Check for persistence mechanisms (scheduled tasks, registry run keys)",
        ],
        "remediation": [
            "Enable PowerShell Constrained Language Mode",
            "Implement PowerShell execution policy (AllSigned or RemoteSigned)",
            "Deploy application control to block unsigned scripts",
            "Enable comprehensive PowerShell logging organisation-wide",
        ],
        "recovery": [
            "Re-image endpoint if payload execution is confirmed",
            "Remove persistence mechanisms",
            "Monitor PowerShell activity for 30 days",
        ],
    },
    "USB Data Theft": {
        "immediate_containment": [
            "Disable USB ports on the endpoint via Group Policy",
            "Recover the USB device if it is still on-premises",
            "Suspend the employee's account pending investigation",
            "Preserve endpoint logs and DLP records",
        ],
        "investigation": [
            "Identify the USB device by serial number via endpoint logs",
            "Enumerate all files copied to the device",
            "Interview the employee regarding the purpose of the transfer",
            "Determine whether the data constitutes a notifiable breach",
        ],
        "remediation": [
            "Enforce USB device control policy via endpoint management",
            "Implement DLP rules for removable media",
            "Whitelist only approved USB devices",
        ],
        "recovery": [
            "Conduct data impact assessment",
            "Engage Legal for breach notification if required",
            "Update removable media policy and training",
        ],
    },
    "Normal": {
        "immediate_containment": ["No containment action required."],
        "investigation": ["No further investigation required."],
        "remediation": ["No remediation action required."],
        "recovery": ["No recovery action required."],
    },
}


# ---------------------------------------------------------------------------
# Known Attack Chain Sequences for Correlation
# ---------------------------------------------------------------------------

ATTACK_CHAIN_SEQUENCES: List[List[str]] = [
    ["Brute Force", "Credential Stuffing", "Lateral Movement", "Data Exfiltration"],
    ["Malware Execution", "Privilege Escalation", "Beaconing C2"],
    ["Impossible Travel", "Off-hours Access", "Data Exfiltration"],
    ["Lateral Movement", "Privilege Escalation", "USB Data Theft"],
    ["Brute Force", "Privilege Escalation", "Data Exfiltration"],
    ["Beaconing C2", "Lateral Movement", "Data Exfiltration"],
    ["Credential Stuffing", "Insider Threat", "Low-and-Slow Exfiltration"],
    ["Malware Execution", "Lateral Movement", "Beaconing C2", "Data Exfiltration"],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def elapsed_str(start_time: float) -> str:
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"


def sanitize_filename(name: str) -> str:
    """Convert a session ID or name to a safe filename."""
    return re.sub(r"[^\w\-]", "_", name)
