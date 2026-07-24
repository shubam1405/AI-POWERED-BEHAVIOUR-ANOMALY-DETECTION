"""
utils.py – Common utilities for the GRU Autoencoder Anomaly Detection Engine.

Provides:
    - set_seed         : Reproducible training across Python / NumPy / PyTorch.
    - get_device       : Automatic CPU/CUDA selection.
    - ensure_dir       : Safe directory creation.
    - elapsed_str      : Human-readable elapsed-time formatter.
    - moving_average   : Smoothed signal helper for plotting.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger("AnomalyDetection.Utils")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for fully reproducible training.

    Parameters
    ----------
    seed : int
        The seed value to apply globally (Python, NumPy, PyTorch CPU & CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic cuDNN behaviour (may reduce performance slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.warning("PyTorch not available – only Python/NumPy seeds applied.")
    logger.debug("Random seed fixed to %d.", seed)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> "torch.device":
    """Return the best available device (CUDA > CPU).

    Parameters
    ----------
    prefer_gpu : bool
        When *True* (default) the function will use CUDA if it is available.
        Set to *False* to force CPU even on GPU machines.

    Returns
    -------
    torch.device
    """
    import torch
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU detected – training on %s.", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("Training on CPU.")
    return device


# ---------------------------------------------------------------------------
# File-system helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and any parent directories) if it does not exist.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Path
        The resolved, absolute ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def elapsed_str(start_time: float) -> str:
    """Return a human-readable string of time elapsed since ``start_time``.

    Parameters
    ----------
    start_time : float
        Value returned by ``time.time()`` at the start of the operation.

    Returns
    -------
    str
        E.g. ``"1m 23s"`` or ``"45s"``.
    """
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    return f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"


# ---------------------------------------------------------------------------
# Signal processing
# ---------------------------------------------------------------------------

def moving_average(values: List[float], window: int = 5) -> List[float]:
    """Compute a simple moving average over *values* with the given *window*.

    Parameters
    ----------
    values : list of float
    window : int
        Number of elements to average over.

    Returns
    -------
    list of float
        Same length as *values*; early elements use a smaller effective window.
    """
    result: List[float] = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        result.append(float(np.mean(values[start : i + 1])))
    return result
