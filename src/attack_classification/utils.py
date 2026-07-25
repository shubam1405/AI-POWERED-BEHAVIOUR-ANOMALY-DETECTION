"""
utils.py – Common utilities for the Attack Classification Engine.

Provides:
    - set_seed       : Reproducible training across Python / NumPy / XGBoost.
    - ensure_dir     : Safe directory creation.
    - elapsed_str    : Human-readable elapsed-time formatter.
    - ATTACK_LABELS  : Canonical ordering of attack class labels.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger("AttackClassification.Utils")


# ---------------------------------------------------------------------------
# Canonical label order — must be consistent across train / eval / inference
# ---------------------------------------------------------------------------

ATTACK_LABELS: List[str] = [
    "Normal",
    "Beaconing C2",
    "Brute Force",
    "Credential Stuffing",
    "Data Exfiltration",
    "Device Spoofing",
    "Impossible Travel",
    "Insider Drift",
    "Insider Threat",
    "Lateral Movement",
    "Low-and-Slow Exfiltration",
    "Malware Execution",
    "Off-hours Access",
    "Privilege Escalation",
    "Suspicious PowerShell",
    "USB Data Theft",
]

# Meta-columns in tabular_features.csv that should NOT be used as model inputs
META_COLUMNS: List[str] = [
    "session_id",
    "employee_id",
    "is_anomalous",
    "attack_type",
    "risk_score",
]

# Columns from reconstruction_scores.csv to merge
SCORE_MERGE_COLUMNS: List[str] = [
    "session_id",
    "reconstruction_error",
    "anomaly_score",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.debug("Random seed fixed to %d.", seed)


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` and parents if they do not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def elapsed_str(start_time: float) -> str:
    """Human-readable elapsed time since ``start_time``."""
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"
