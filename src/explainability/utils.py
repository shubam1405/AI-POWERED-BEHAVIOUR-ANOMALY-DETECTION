"""
utils.py – Shared knowledge base and helpers for the Explainability Engine.

Contains:
    - MITRE ATT&CK mapping for all 16 classes
    - Feature category groupings (8 categories)
    - Feature human-readable descriptions
    - Severity scoring logic
    - SHAP impact threshold labels
    - Attack-specific NL context strings
    - Investigation step recommendations
    - Common filesystem / timing helpers
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Explainability.Utils")

# ---------------------------------------------------------------------------
# MITRE ATT&CK Mapping
# ---------------------------------------------------------------------------

MITRE_MAPPING: Dict[str, Optional[Dict[str, str]]] = {
    "Normal": None,
    "Beaconing C2": {
        "tactic": "Command and Control",
        "technique_id": "T1071",
        "technique": "Application Layer Protocol",
    },
    "Brute Force": {
        "tactic": "Credential Access",
        "technique_id": "T1110",
        "technique": "Brute Force",
    },
    "Credential Stuffing": {
        "tactic": "Credential Access",
        "technique_id": "T1110.004",
        "technique": "Credential Stuffing",
    },
    "Data Exfiltration": {
        "tactic": "Exfiltration",
        "technique_id": "T1048",
        "technique": "Exfiltration Over Alternative Protocol",
    },
    "Device Spoofing": {
        "tactic": "Defense Evasion",
        "technique_id": "T1036.005",
        "technique": "Match Legitimate Name or Location",
    },
    "Impossible Travel": {
        "tactic": "Initial Access",
        "technique_id": "T1078",
        "technique": "Valid Accounts",
    },
    "Insider Drift": {
        "tactic": "Defense Evasion",
        "technique_id": "T1036",
        "technique": "Masquerading",
    },
    "Insider Threat": {
        "tactic": "Collection",
        "technique_id": "T1074",
        "technique": "Data Staged",
    },
    "Lateral Movement": {
        "tactic": "Lateral Movement",
        "technique_id": "T1021",
        "technique": "Remote Services",
    },
    "Low-and-Slow Exfiltration": {
        "tactic": "Exfiltration",
        "technique_id": "T1029",
        "technique": "Scheduled Transfer",
    },
    "Malware Execution": {
        "tactic": "Execution",
        "technique_id": "T1059",
        "technique": "Command and Scripting Interpreter",
    },
    "Off-hours Access": {
        "tactic": "Persistence",
        "technique_id": "T1078.003",
        "technique": "Local Accounts",
    },
    "Privilege Escalation": {
        "tactic": "Privilege Escalation",
        "technique_id": "T1068",
        "technique": "Exploitation for Privilege Escalation",
    },
    "Suspicious PowerShell": {
        "tactic": "Execution",
        "technique_id": "T1059.001",
        "technique": "PowerShell",
    },
    "USB Data Theft": {
        "tactic": "Exfiltration",
        "technique_id": "T1052.001",
        "technique": "Exfiltration over USB",
    },
}

# ---------------------------------------------------------------------------
# Feature Categories
# ---------------------------------------------------------------------------

FEATURE_CATEGORIES: Dict[str, List[str]] = {
    "Authentication": [
        "failed_login_count",
        "authentication_attempts",
        "failed_login_ratio",
        "password_resets",
        "mfa_bypassed",
        "login_hour",
    ],
    "Device": [
        "device_deviation",
        "usb_activity",
        "removable_media_count",
        "new_device_flag",
        "device_change_count",
    ],
    "Location": [
        "location_deviation",
        "vpn_used",
        "unique_source_ips",
        "geolocation_anomaly",
        "impossible_travel_flag",
    ],
    "Resource": [
        "resource_access_deviation",
        "files_downloaded",
        "files_uploaded",
        "database_query_count",
        "sensitive_files_accessed",
        "total_download_bytes",
        "total_upload_bytes",
    ],
    "Network": [
        "network_connections",
        "external_connections",
        "upload_download_ratio",
        "bytes_sent",
        "bytes_received",
        "unique_destinations",
        "dns_queries",
    ],
    "Command": [
        "powershell_executions",
        "suspicious_process_count",
        "beaconing_event_count",
        "script_executions",
        "cmd_executions",
        "encoded_commands",
    ],
    "Temporal": [
        "session_start_hour",
        "session_duration_minutes",
        "after_hours_flag",
        "weekend_access",
        "time_since_last_login",
    ],
    "Session": [
        "event_sequence_length",
        "reconstruction_error",
        "anomaly_score",
        "risk_score",
        "event_count",
        "unique_event_types",
    ],
}

# Reverse lookup: feature → category
_FEATURE_TO_CATEGORY: Dict[str, str] = {}
for _cat, _feats in FEATURE_CATEGORIES.items():
    for _f in _feats:
        _FEATURE_TO_CATEGORY[_f] = _cat


def get_feature_category(feature: str) -> str:
    """Return the category for a given feature name (default 'Session')."""
    return _FEATURE_TO_CATEGORY.get(feature, "Session")


# ---------------------------------------------------------------------------
# Feature Human-Readable Descriptions
# ---------------------------------------------------------------------------

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "upload_download_ratio":     "ratio of uploaded to downloaded data",
    "files_downloaded":          "number of files downloaded",
    "external_connections":      "number of outbound external connections",
    "powershell_executions":     "number of PowerShell executions",
    "total_upload_bytes":        "total bytes uploaded",
    "unique_source_ips":         "number of distinct source IP addresses",
    "session_start_hour":        "hour of day the session started",
    "suspicious_process_count":  "number of suspicious processes detected",
    "beaconing_event_count":     "number of beaconing events detected",
    "location_deviation":        "deviation from typical login location",
    "login_hour":                "hour of day when login occurred",
    "total_download_bytes":      "total bytes downloaded",
    "database_query_count":      "number of database queries executed",
    "authentication_attempts":   "number of authentication attempts",
    "failed_login_count":        "number of failed login attempts",
    "failed_login_ratio":        "ratio of failed to total login attempts",
    "device_deviation":          "deviation from typical device fingerprint",
    "vpn_used":                  "whether a VPN was used for this session",
    "network_connections":       "total number of network connections",
    "event_sequence_length":     "length of the event sequence",
    "resource_access_deviation": "deviation from typical resource access patterns",
    "anomaly_score":             "GRU autoencoder anomaly score",
    "reconstruction_error":      "GRU autoencoder reconstruction error",
    "risk_score":                "composite behavioural risk score",
    "session_duration_minutes":  "duration of the session in minutes",
    "usb_activity":              "USB device activity detected",
    "removable_media_count":     "number of removable media interactions",
    "files_uploaded":            "number of files uploaded",
    "sensitive_files_accessed":  "number of sensitive files accessed",
}


def get_feature_description(feature: str) -> str:
    """Return a human-readable description for a feature name."""
    return FEATURE_DESCRIPTIONS.get(
        feature, feature.replace("_", " ")
    )


# ---------------------------------------------------------------------------
# SHAP Impact Labels
# ---------------------------------------------------------------------------

IMPACT_THRESHOLDS: List[Tuple[float, str]] = [
    (0.30,  "High Positive"),
    (0.15,  "Medium Positive"),
    (0.05,  "Low Positive"),
    (-0.05, "Negligible"),
    (-0.15, "Low Negative"),
    (-0.30, "Medium Negative"),
]
_HIGH_NEG = "High Negative"


def get_impact_label(shap_value: float) -> str:
    """Convert a SHAP value to a human-readable impact label."""
    for threshold, label in IMPACT_THRESHOLDS:
        if shap_value >= threshold:
            return label
    return _HIGH_NEG


# ---------------------------------------------------------------------------
# Severity Classification
# ---------------------------------------------------------------------------

def compute_severity(
    confidence: float,
    anomaly_score: float,
    risk_score: float,
    risk_score_max: float = 100.0,
) -> str:
    """Derive a severity level from model outputs.

    Weighted formula::

        score = confidence * 0.35 + anomaly_score * 0.35 + norm(risk_score) * 0.30

    Returns
    -------
    str : one of ``"Critical"``, ``"High"``, ``"Medium"``, ``"Low"``
    """
    norm_risk = min(risk_score / risk_score_max, 1.0)
    weighted = confidence * 0.35 + anomaly_score * 0.35 + norm_risk * 0.30
    if weighted >= 0.80:
        return "Critical"
    if weighted >= 0.60:
        return "High"
    if weighted >= 0.35:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Attack-Specific NL Context Strings
# ---------------------------------------------------------------------------

ATTACK_NL_CONTEXT: Dict[str, str] = {
    "Beaconing C2": (
        "The session shows signs of Command and Control communication, "
        "where malware periodically contacts a remote server."
    ),
    "Brute Force": (
        "Multiple rapid authentication failures suggest an automated "
        "credential brute-force attack."
    ),
    "Credential Stuffing": (
        "Authentication patterns are consistent with automated credential "
        "stuffing using previously leaked username/password pairs."
    ),
    "Data Exfiltration": (
        "The session exhibits large-scale outbound data transfer and "
        "abnormal file access patterns typical of data exfiltration."
    ),
    "Device Spoofing": (
        "The session device fingerprint does not match the user's known "
        "devices, suggesting potential device identity spoofing."
    ),
    "Impossible Travel": (
        "Authentication events originated from geographically impossible "
        "locations within an impossibly short time window."
    ),
    "Insider Drift": (
        "The user's behaviour has gradually shifted away from their "
        "established baseline, which may indicate insider activity over time."
    ),
    "Insider Threat": (
        "The session shows access patterns — such as bulk data staging — "
        "that are consistent with malicious insider activity."
    ),
    "Lateral Movement": (
        "The session shows connections to multiple internal systems that "
        "are not typical for this user, suggesting lateral movement."
    ),
    "Low-and-Slow Exfiltration": (
        "Data is being exfiltrated in small increments over an extended "
        "period, consistent with low-and-slow exfiltration techniques."
    ),
    "Malware Execution": (
        "Suspicious process execution patterns and script activity indicate "
        "possible malware execution on the user's device."
    ),
    "Off-hours Access": (
        "The session occurred outside normal business hours with unusual "
        "resource access patterns."
    ),
    "Privilege Escalation": (
        "The session shows attempts to access resources beyond the user's "
        "authorised privilege level."
    ),
    "Suspicious PowerShell": (
        "An unusually high volume of PowerShell executions — potentially "
        "with encoded or obfuscated commands — was observed."
    ),
    "USB Data Theft": (
        "Significant data transfer to a removable USB device was detected "
        "during this session."
    ),
    "Normal": (
        "No significant anomalies were detected. "
        "The session conforms to the user's established baseline."
    ),
}

# ---------------------------------------------------------------------------
# Recommended Investigation Steps
# ---------------------------------------------------------------------------

INVESTIGATION_STEPS: Dict[str, List[str]] = {
    "Beaconing C2": [
        "Inspect outbound network connections for periodic intervals (beaconing signatures)",
        "Review DNS query logs for suspicious or newly registered domains",
        "Scan the endpoint for malware or remote access tools (RATs)",
        "Check firewall and proxy logs for C2 IP addresses against threat intel feeds",
        "Isolate the device and perform a full forensic investigation",
    ],
    "Brute Force": [
        "Review authentication logs for the originating IP address",
        "Determine whether any login attempts were successful",
        "Block the source IP and apply temporary account lockout",
        "Verify whether multi-factor authentication is enforced for this account",
        "Check for credential exposure in known breach databases",
    ],
    "Credential Stuffing": [
        "Cross-reference failed authentication usernames against known breach datasets",
        "Review source IPs for proxy or Tor exit node characteristics",
        "Enforce MFA or CAPTCHA on the affected authentication endpoint",
        "Check whether any matching credentials resulted in successful logins",
        "Notify affected users and prompt password reset",
    ],
    "Data Exfiltration": [
        "Review DLP alerts and outbound data transfer logs for this session",
        "Inspect email attachments and cloud storage uploads during the session window",
        "Verify business justification for large file downloads with the user's manager",
        "Check network egress for unusual destinations or transfer volumes",
        "Preserve a forensic copy of accessed files and network traffic",
    ],
    "Device Spoofing": [
        "Compare the session device fingerprint against the user's registered devices",
        "Verify the physical location of the device at the time of the session",
        "Review recent device registration or enrollment events for this account",
        "Inspect the session for other anomalous indicators (unusual IPs, agents)",
        "Revoke device trust and require re-authentication from a known device",
    ],
    "Impossible Travel": [
        "Confirm the two geographic locations and calculate travel feasibility",
        "Determine whether a VPN or proxy was in use for either session",
        "Contact the user directly to verify whether both sessions are legitimate",
        "Check whether account credentials may have been shared or stolen",
        "Temporarily suspend the account pending investigation if compromise is suspected",
    ],
    "Insider Drift": [
        "Review the user's access history over the past 30–90 days for gradual shifts",
        "Compare current resource access against the user's established peer group",
        "Interview the user or their manager about recent role or responsibility changes",
        "Determine whether the behaviour aligns with any recent organisational changes",
        "Flag the account for enhanced monitoring if no legitimate explanation is found",
    ],
    "Insider Threat": [
        "Review all files accessed and downloaded during the session",
        "Check for bulk data staging activity on local or shared drives",
        "Verify whether the user recently resigned, was under performance review, or had system access changed",
        "Preserve all relevant logs and evidence for HR and legal review",
        "Escalate to the insider threat management team",
    ],
    "Lateral Movement": [
        "Map all internal systems accessed during the session",
        "Identify which systems are outside the user's normal access scope",
        "Review remote desktop, SMB, and SSH connection logs",
        "Check whether administrative credentials were used",
        "Contain the affected systems and review for further compromise",
    ],
    "Low-and-Slow Exfiltration": [
        "Analyse cumulative outbound data volumes over the past 7–30 days",
        "Review file access logs for sensitive data being accessed in small increments",
        "Check for scheduled tasks or scripts that automate small data transfers",
        "Correlate the activity with any known insider threat indicators",
        "Implement data egress controls and alerting for low-volume repeated transfers",
    ],
    "Malware Execution": [
        "Collect and analyse suspicious process execution logs and parent-child process trees",
        "Submit any unknown executables to a sandboxing environment for dynamic analysis",
        "Review script execution logs (PowerShell, VBScript, Python) for encoded commands",
        "Isolate the endpoint from the network immediately",
        "Run a full endpoint detection and response (EDR) scan",
    ],
    "Off-hours Access": [
        "Confirm whether off-hours access was pre-approved or expected",
        "Review the resources accessed during the off-hours session",
        "Contact the user or on-call manager to verify legitimacy",
        "Check whether the session originated from an expected location and device",
        "Review for correlation with other anomalous events in the same timeframe",
    ],
    "Privilege Escalation": [
        "Review which privileged resources or accounts were accessed",
        "Check for recent changes to the user's group memberships or roles",
        "Inspect sudo, UAC, or admin access logs on affected systems",
        "Determine whether any exploits or misconfigurations were leveraged",
        "Revoke any unauthorised privilege grants immediately",
    ],
    "Suspicious PowerShell": [
        "Review the full PowerShell command history for the session",
        "Check for encoded or obfuscated commands (Base64, -EncodedCommand flags)",
        "Determine whether any scripts were downloaded from the internet (DownloadString, IEX)",
        "Scan the endpoint for persistence mechanisms (scheduled tasks, registry keys)",
        "Review PowerShell logging (Script Block Logging, Module Logging) if enabled",
    ],
    "USB Data Theft": [
        "Identify the USB device by serial number and check it against the asset register",
        "Review the files copied to the USB device using DLP or endpoint logs",
        "Physically locate and secure the USB device if possible",
        "Interview the user regarding the purpose of the data transfer",
        "Enforce USB device control policies and review current DLP rules",
    ],
    "Normal": [
        "No immediate action required — session appears within normal behavioural baseline.",
    ],
}


# ---------------------------------------------------------------------------
# Filesystem and timing helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory and parents if not present."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def elapsed_str(start_time: float) -> str:
    """Human-readable elapsed time since *start_time*."""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
