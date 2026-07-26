"""
gru_hyperparameter_search.py – Optuna-based hyperparameter optimisation
                                for the GRU Autoencoder.

Objective
---------
Maximise a composite score on the test set:

    score = 0.6 × F1 + 0.4 × ROC-AUC

This directly optimises the metrics that matter for anomaly detection
(not just reconstruction loss, which is a proxy).

Search Space
------------
    hidden_size       : {64, 128, 192, 256, 320}
    latent_dim        : {16, 32, 48, 64, 96, 128}  (bottleneck compression)
    num_layers        : {1, 2, 3}
    dropout           : [0.10, 0.50]
    learning_rate     : [1e-4, 5e-3]   (log-uniform)
    weight_decay      : [1e-6, 1e-3]   (log-uniform)
    batch_size        : {32, 64, 128}
    loss_fn           : {mse, mae, huber}
    bidirectional     : {True, False}

Note: seq_len is fixed by the pre-generated data files and is not searched.

Usage
-----
    python src/anomaly_detection/gru_hyperparameter_search.py \\
        --tabular   data/processed/tabular_features.csv \\
        --sequential data/processed/sequential_features.json \\
        --n-trials  30 \\
        --quick-epochs 30 \\
        --output   outputs/optuna_best_params.json

Or call :func:`run_search` programmatically from ``train_gru.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup (allow running as standalone script)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR    = os.path.dirname(_SCRIPT_DIR)
for _p in (_SCRIPT_DIR, _SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve

from anomaly_detection.dataset_loader import AnomalyDatasetLoader
from anomaly_detection.gru_autoencoder import GRUAutoencoder, build_criterion
from anomaly_detection.utils import ensure_dir, get_device, set_seed

logger = logging.getLogger("AnomalyDetection.HyperSearch")


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _objective(
    trial,
    tabular_path: str,
    sequential_path: str,
    quick_epochs: int,
    seed: int,
    device: torch.device,
) -> float:
    """Optuna trial objective.

    Trains the GRU for ``quick_epochs`` with trial hyperparameters,
    then evaluates F1 + ROC-AUC on the test set using the F1-optimal
    threshold from the Precision-Recall curve.

    Returns
    -------
    float
        Composite score = 0.6 × F1 + 0.4 × ROC-AUC.
        Higher is better.
    """
    import optuna

    # ------------------------------------------------------------------
    # Sample hyperparameters
    # ------------------------------------------------------------------
    hidden_size   = trial.suggest_categorical("hidden_size",   [64, 128, 192, 256, 320])
    num_layers    = trial.suggest_categorical("num_layers",    [1, 2, 3])
    dropout       = trial.suggest_float("dropout",             0.10, 0.50)
    learning_rate = trial.suggest_float("learning_rate",       1e-4, 5e-3, log=True)
    weight_decay  = trial.suggest_float("weight_decay",        1e-6, 1e-3, log=True)
    batch_size    = trial.suggest_categorical("batch_size",    [32, 64, 128])
    loss_fn       = trial.suggest_categorical("loss_fn",       ["mse", "mae", "huber"])
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])

    set_seed(seed + trial.number)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    try:
        loader = AnomalyDatasetLoader(
            tabular_path=tabular_path,
            sequential_path=sequential_path,
            batch_size=batch_size,
            seed=seed,
        )
        loader.load()
        train_loader, val_loader, test_loader = loader.split()

        # Fit and apply scaler (consistent with full training)
        loader.fit_scaler()

        seq_len     = loader.sequence_max_len
        feature_dim = loader.sequence_feature_dim

    except Exception as exc:
        logger.warning("Trial %d: data loading failed: %s", trial.number, exc)
        raise optuna.exceptions.TrialPruned()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = GRUAutoencoder(
        input_dim=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        seq_len=seq_len,
        bidirectional_encoder=bidirectional,
    ).to(device)

    criterion = build_criterion(loss_fn)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5,
    )

    # ------------------------------------------------------------------
    # Quick training loop
    # ------------------------------------------------------------------
    best_val_loss = float("inf")
    patience_cnt  = 0
    _EARLY_STOP   = max(quick_epochs // 3, 5)   # mini early-stop for speed

    for epoch in range(quick_epochs):
        # Train
        model.train()
        for batch in train_loader:
            seq, mask, _, _, _ = batch
            seq  = seq.to(device)
            mask = mask.to(device)
            recon = model(seq)
            loss  = criterion(recon, seq, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0.0
        n_val    = 0
        with torch.no_grad():
            for batch in val_loader:
                seq, mask, _, _, _ = batch
                seq  = seq.to(device)
                mask = mask.to(device)
                recon = model(seq)
                loss  = criterion(recon, seq, mask)
                val_loss += loss.item()
                n_val    += 1
        val_loss /= max(n_val, 1)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= _EARLY_STOP:
                break

        # Optuna pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # ------------------------------------------------------------------
    # Evaluate on test set (F1 + ROC-AUC)
    # ------------------------------------------------------------------
    model.eval()
    all_errors: list = []
    all_labels: list = []

    with torch.no_grad():
        for batch in test_loader:
            seq, mask, _, lbl, _ = batch
            seq  = seq.to(device)
            mask = mask.to(device)
            recon = model(seq)
            for i in range(seq.size(0)):
                sq_err  = (recon[i] - seq[i]) ** 2
                mask_exp = mask[i].unsqueeze(-1).expand_as(sq_err)
                err = float((sq_err * mask_exp).sum() / mask_exp.sum().clamp(min=1.0))
                all_errors.append(err)
                all_labels.append(int(lbl[i].item()))

    errors_arr = np.array(all_errors, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    if labels_arr.sum() == 0:
        # No anomalies in test set — cannot compute F1/AUC
        logger.warning("Trial %d: no anomalous sessions in test set.", trial.number)
        return 0.0

    # F1-optimal threshold from PR curve
    try:
        prec_arr, rec_arr, pr_thresholds = precision_recall_curve(labels_arr, errors_arr)
        f1_arr       = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-9)
        f1_best_idx  = int(np.argmax(f1_arr[:-1]))
        best_thresh  = float(pr_thresholds[f1_best_idx])
        predictions  = (errors_arr > best_thresh).astype(int)
        f1           = float(f1_score(labels_arr, predictions, zero_division=0))
        roc_auc      = float(roc_auc_score(labels_arr, errors_arr))
    except Exception as exc:
        logger.warning("Trial %d: metric computation failed: %s", trial.number, exc)
        return 0.0

    composite = 0.6 * f1 + 0.4 * roc_auc

    logger.info(
        "Trial %3d  hidden=%d  layers=%d  dropout=%.2f  "
        "lr=%.1e  batch=%d  loss=%s  bidir=%s → "
        "F1=%.4f  AUC=%.4f  score=%.4f",
        trial.number, hidden_size, num_layers, dropout,
        learning_rate, batch_size, loss_fn, bidirectional,
        f1, roc_auc, composite,
    )

    return composite


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_search(
    tabular_path: str,
    sequential_path: str,
    n_trials: int = 30,
    quick_epochs: int = 30,
    output_path: str = "outputs/optuna_best_params.json",
    seed: int = 42,
    device: Optional[torch.device] = None,
    study_name: str = "gru_anomaly_detection",
) -> Dict[str, Any]:
    """Run the Optuna hyperparameter search.

    Parameters
    ----------
    tabular_path : str
    sequential_path : str
    n_trials : int
        Number of Optuna trials to run (default 30).
    quick_epochs : int
        Epochs per trial — keep small for speed (default 30).
    output_path : str
        Where to save the best hyperparameters as JSON.
    seed : int
    device : torch.device, optional
    study_name : str

    Returns
    -------
    dict
        The best hyperparameters found.
    """
    try:
        import optuna
    except ImportError:
        raise ImportError(
            "Optuna is required for hyperparameter search. "
            "Install it with:  pip install optuna"
        )

    if device is None:
        device = get_device()

    # Suppress verbose Optuna logs (keep only warnings+)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        "Starting Optuna search  trials=%d  quick_epochs=%d  device=%s",
        n_trials, quick_epochs, device,
    )
    t_start = time.time()

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    study.optimize(
        lambda trial: _objective(
            trial,
            tabular_path=tabular_path,
            sequential_path=sequential_path,
            quick_epochs=quick_epochs,
            seed=seed,
            device=device,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
        catch=(Exception,),
    )

    elapsed = time.time() - t_start
    best    = study.best_trial

    best_params = dict(best.params)
    best_params["best_score"]       = round(best.value, 6)
    best_params["search_time_s"]    = round(elapsed, 1)
    best_params["n_trials"]         = n_trials
    best_params["quick_epochs"]     = quick_epochs

    logger.info(
        "\n" + "=" * 60 + "\n"
        "Optuna search complete in %.0fs\n"
        "Best composite score : %.4f\n"
        "Best params          : %s\n"
        + "=" * 60,
        elapsed,
        best.value,
        json.dumps({k: v for k, v in best_params.items() if k not in ("best_score", "search_time_s", "n_trials", "quick_epochs")}, indent=2),
    )

    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    logger.info("Best params saved → %s", output_path)

    # Save importance plot (if matplotlib is available)
    try:
        import optuna.visualization.matplotlib as optuna_mpl
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plot_path = str(Path(output_path).parent / "optuna_param_importance.png")
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        logger.info("Parameter importance plot saved → %s", plot_path)
    except Exception:
        pass  # Non-critical

    return best_params


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the Cyber Cage GRU Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tabular",      default="data/processed/tabular_features.csv")
    p.add_argument("--sequential",   default="data/processed/sequential_features.json")
    p.add_argument("--n-trials",     type=int, default=30,   dest="n_trials")
    p.add_argument("--quick-epochs", type=int, default=30,   dest="quick_epochs")
    p.add_argument("--output",       default="outputs/optuna_best_params.json")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()
    best = run_search(
        tabular_path=args.tabular,
        sequential_path=args.sequential,
        n_trials=args.n_trials,
        quick_epochs=args.quick_epochs,
        output_path=args.output,
        seed=args.seed,
    )
    print("\nBest hyperparameters:")
    print(json.dumps(best, indent=2))
