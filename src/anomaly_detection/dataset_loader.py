"""
dataset_loader.py – Data loading, alignment, splitting, and PyTorch DataLoaders.

Design
------
The feature engineering pipeline produces two positionally-aligned output files:

    data/processed/tabular_features.csv      – one row per session
    data/processed/sequential_features.json  – list of (50 × 21) sequences

Row *i* in the CSV corresponds to sequence *i* in the JSON.

Split strategy
--------------
* **Train**      : normal sessions only (is_anomalous == 0) – 70 %
* **Validation** : normal sessions only                      – 15 %
* **Test**       : ALL sessions (normal + anomalous)         – 15 %

Normalisation
-------------
Call ``fit_scaler()`` after ``split()`` to fit a StandardScaler on the
training sequences and apply it to all splits.  The scaler is fit on
normal training sessions only (consistent with GRU's one-class training).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("AnomalyDetection.DatasetLoader")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns in tabular_features.csv that are NOT numerical feature inputs.
_META_COLUMNS: Tuple[str, ...] = (
    "session_id",
    "employee_id",
    "is_anomalous",
    "attack_type",
    "risk_score",
)

# Columns that must be cast to float for model input.
_CATEGORICAL_ENCODE: Tuple[str, ...] = (
    "browser",
    "operating_system",
    "department",
    "login_method",
    "first_event_type",
    "last_event_type",
)


# ---------------------------------------------------------------------------
# Data-class
# ---------------------------------------------------------------------------

@dataclass
class SessionRecord:
    """All data associated with a single session."""

    session_id: str
    employee_id: str
    is_anomalous: int                    # 0 = normal, 1 = anomalous
    attack_type: str                     # "None" for normal sessions
    risk_score: float
    tabular_features: np.ndarray         # shape (num_tabular_features,)
    sequence: np.ndarray                 # shape (max_len, feature_dim)
    mask: np.ndarray                     # shape (max_len,)  1=real, 0=pad


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class SessionDataset(Dataset):
    """PyTorch Dataset wrapping a list of :class:`SessionRecord` objects.

    Each item returned is a tuple:
    ``(sequence_tensor, mask_tensor, tabular_tensor, label, session_id)``

    Parameters
    ----------
    records : list of SessionRecord
    """

    def __init__(self, records: List[SessionRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str
    ]:
        rec = self.records[idx]
        seq = torch.tensor(rec.sequence, dtype=torch.float32)
        msk = torch.tensor(rec.mask, dtype=torch.float32)
        tab = torch.tensor(rec.tabular_features, dtype=torch.float32)
        lbl = torch.tensor(rec.is_anomalous, dtype=torch.long)
        return seq, msk, tab, lbl, rec.session_id


# ---------------------------------------------------------------------------
# Main loader class
# ---------------------------------------------------------------------------

class AnomalyDatasetLoader:
    """Loads, validates, aligns, normalises, and splits the Cyber Cage feature files.

    Parameters
    ----------
    tabular_path : str
        Path to ``tabular_features.csv`` produced by FeatureEngineer.
    sequential_path : str
        Path to ``sequential_features.json`` produced by SequenceBuilder.
    train_ratio : float
        Fraction of *normal* sessions to use for training (default 0.70).
    val_ratio : float
        Fraction of *normal* sessions to use for validation (default 0.15).
    seed : int
        Random seed for reproducible splits.
    batch_size : int
        DataLoader batch size.
    num_workers : int
        DataLoader worker count.

    Raises
    ------
    FileNotFoundError
        If either input file does not exist.
    ValueError
        If the number of records in CSV and JSON do not match.
    """

    def __init__(
        self,
        tabular_path: str = "data/processed/tabular_features.csv",
        sequential_path: str = "data/processed/sequential_features.json",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self.tabular_path    = Path(tabular_path)
        self.sequential_path = Path(sequential_path)
        self.train_ratio     = train_ratio
        self.val_ratio       = val_ratio
        self.seed            = seed
        self.batch_size      = batch_size
        self.num_workers     = num_workers

        self._validate_paths()
        self.records: List[SessionRecord]      = []
        self.feature_columns: List[str]        = []
        self.sequence_max_len: int             = 0
        self.sequence_feature_dim: int         = 0

        # Populated after split()
        self._train_idx: List[int] = []
        self._val_idx:   List[int] = []
        self._test_idx:  List[int] = []

        # Populated after fit_scaler()
        self._scaler: Optional[StandardScaler] = None

        # Populated after load()
        self._category_maps: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "AnomalyDatasetLoader":
        """Load, validate, and align both input files.

        Returns
        -------
        self : AnomalyDatasetLoader
            Fluent interface – allows ``loader.load().split()``.
        """
        logger.info("Loading tabular features from %s …", self.tabular_path)
        tabular_rows, raw_feature_cols = self._load_tabular()

        logger.info("Loading sequential features from %s …", self.sequential_path)
        seq_data = self._load_sequential()

        sequences: List[List[List[float]]] = seq_data["sequences"]
        masks: List[List[float]]           = seq_data["masks"]
        self.sequence_max_len              = seq_data["max_len"]
        self.sequence_feature_dim          = seq_data["feature_dim"]

        self._validate_alignment(len(tabular_rows), len(sequences))

        # Build category maps for one-hot / ordinal encoding
        self._build_category_maps(tabular_rows, raw_feature_cols)

        # Determine final feature columns (after encoding)
        self.feature_columns = self._resolve_feature_columns(raw_feature_cols)

        logger.info("Building %d SessionRecord objects …", len(tabular_rows))
        for i, row in enumerate(tabular_rows):
            tab_vec = self._row_to_vector(row, raw_feature_cols)
            seq_arr = np.array(sequences[i], dtype=np.float32)
            msk_arr = np.array(masks[i], dtype=np.float32)

            record = SessionRecord(
                session_id=row["session_id"],
                employee_id=row["employee_id"],
                is_anomalous=int(float(row["is_anomalous"])),
                attack_type=row.get("attack_type", "None"),
                risk_score=float(row.get("risk_score", 0.0)),
                tabular_features=tab_vec,
                sequence=seq_arr,
                mask=msk_arr,
            )
            self.records.append(record)

        n_normal = sum(1 for r in self.records if r.is_anomalous == 0)
        n_anom   = len(self.records) - n_normal
        logger.info(
            "Loaded %d sessions  (normal=%d, anomalous=%d).",
            len(self.records), n_normal, n_anom,
        )
        return self

    def split(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Split records into train / val / test DataLoaders.

        Train and validation sets contain **only normal sessions**.
        The test set contains **all sessions** (for evaluation).

        Split indices are stored as ``_train_idx``, ``_val_idx``, ``_test_idx``
        so that :meth:`fit_scaler` can use them without arguments.

        Returns
        -------
        train_loader, val_loader, test_loader : DataLoader
        """
        if not self.records:
            raise RuntimeError("Call load() before split().")

        all_idx = list(range(len(self.records)))

        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        # Stratified split to preserve class ratio in test set
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=test_ratio,
            random_state=self.seed,
            stratify=[self.records[i].is_anomalous for i in all_idx],
        )

        # Keep only normal sessions for train+val
        train_val_normal = [i for i in train_val_idx if self.records[i].is_anomalous == 0]
        relative_val     = self.val_ratio / (self.train_ratio + self.val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_normal,
            test_size=relative_val,
            random_state=self.seed,
        )

        # Store indices for scaler fitting
        self._train_idx = train_idx
        self._val_idx   = val_idx
        self._test_idx  = test_idx

        logger.info(
            "Split → train (normal)=%d  val (normal)=%d  test (all)=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )

        full_ds = SessionDataset(self.records)

        def _loader(indices: List[int], shuffle: bool) -> DataLoader:
            return DataLoader(
                Subset(full_ds, indices),
                batch_size=self.batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                pin_memory=False,
            )

        return (
            _loader(train_idx, shuffle=True),
            _loader(val_idx,   shuffle=False),
            _loader(test_idx,  shuffle=False),
        )

    def fit_scaler(self) -> StandardScaler:
        """Fit a StandardScaler on training sequences and apply to all splits.

        The scaler is fit on real (non-padded) time-steps from normal
        training sessions only.  It is then applied in-place to all session
        sequences (train, val, test) so the GRU sees consistent scaling.

        Must be called **after** :meth:`split`.

        Returns
        -------
        StandardScaler
            The fitted scaler (also stored as ``self._scaler``).

        Raises
        ------
        RuntimeError
            If :meth:`split` has not been called yet.
        """
        if not self._train_idx:
            raise RuntimeError("Call split() before fit_scaler().")

        # Collect all real time-steps from training (normal) sequences
        train_steps: List[np.ndarray] = []
        for idx in self._train_idx:
            rec  = self.records[idx]
            mask = rec.mask                    # (seq_len,)
            seq  = rec.sequence                # (seq_len, feature_dim)
            real = seq[mask == 1]              # (n_real, feature_dim)
            if real.shape[0] > 0:
                train_steps.append(real)

        if not train_steps:
            raise RuntimeError("No real time-steps found in training records.")

        all_steps = np.vstack(train_steps)    # (total_steps, feature_dim)
        scaler    = StandardScaler()
        scaler.fit(all_steps)

        logger.info(
            "StandardScaler fitted on %d time-steps from %d training sessions.",
            all_steps.shape[0], len(self._train_idx),
        )

        # Apply transform to ALL session sequences in-place
        for rec in self.records:
            rec.sequence = scaler.transform(rec.sequence).astype(np.float32)

        self._scaler = scaler
        logger.info("Scaler applied to all %d session sequences.", len(self.records))
        return scaler

    def get_scaler(self) -> Optional[StandardScaler]:
        """Return the fitted scaler (None if :meth:`fit_scaler` has not been called)."""
        return self._scaler

    def get_split_indices(self) -> Tuple[List[int], List[int], List[int]]:
        """Return (train_idx, val_idx, test_idx) from the last split() call."""
        return self._train_idx, self._val_idx, self._test_idx

    def get_all_records(self) -> List[SessionRecord]:
        """Return the full list of :class:`SessionRecord` objects after load()."""
        return self.records

    def num_tabular_features(self) -> int:
        """Number of tabular feature dimensions after encoding."""
        if not self.records:
            raise RuntimeError("Call load() first.")
        return int(self.records[0].tabular_features.shape[0])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_paths(self) -> None:
        for p in (self.tabular_path, self.sequential_path):
            if not p.exists():
                raise FileNotFoundError(f"Required file not found: {p}")

    def _load_tabular(self) -> Tuple[List[Dict[str, str]], List[str]]:
        """Return (rows, feature_columns) where feature_columns excludes meta."""
        rows: List[Dict[str, str]] = []
        with open(self.tabular_path, newline="", encoding="utf-8") as f:
            reader    = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                if all(v in ("", None) for v in row.values()):
                    continue
                rows.append(dict(row))

        feature_cols = [c for c in fieldnames if c not in _META_COLUMNS]
        logger.info("Tabular: %d rows, %d feature columns.", len(rows), len(feature_cols))
        return rows, feature_cols

    def _load_sequential(self) -> Dict:
        with open(self.sequential_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        required = {"sequences", "masks", "max_len", "feature_dim"}
        missing  = required - set(data.keys())
        if missing:
            raise ValueError(f"sequential_features.json is missing keys: {missing}")
        return data

    def _validate_alignment(self, n_tab: int, n_seq: int) -> None:
        if n_tab != n_seq:
            raise ValueError(
                f"Alignment error: {n_tab} tabular rows vs {n_seq} sequences. "
                "Both files must be generated from the same pipeline run."
            )
        logger.info("Alignment check passed: %d sessions in both files.", n_tab)

    def _build_category_maps(
        self,
        rows: List[Dict[str, str]],
        feature_cols: List[str],
    ) -> None:
        """Build ordinal encoding maps for string categorical columns."""
        for col in feature_cols:
            if col in _CATEGORICAL_ENCODE:
                unique_vals = sorted({row[col] for row in rows if row.get(col)})
                self._category_maps[col] = {v: i for i, v in enumerate(unique_vals)}
                logger.debug("Category map for '%s': %d values.", col, len(unique_vals))

    def _resolve_feature_columns(self, raw_cols: List[str]) -> List[str]:
        """Return the final list of feature column names (post-encoding)."""
        return raw_cols  # Ordinal encoding keeps same column names

    def _row_to_vector(
        self,
        row: Dict[str, str],
        feature_cols: List[str],
    ) -> np.ndarray:
        """Convert a CSV row dict into a float32 numpy feature vector."""
        values: List[float] = []
        for col in feature_cols:
            raw = row.get(col, "0") or "0"
            if col in self._category_maps:
                encoded = float(self._category_maps[col].get(raw, -1))
                values.append(encoded)
            else:
                try:
                    values.append(float(raw))
                except (ValueError, TypeError):
                    values.append(0.0)
        return np.array(values, dtype=np.float32)
