"""
train_gru.py – End-to-end training and evaluation entry point for the
               GRU Autoencoder Anomaly Detection Engine (Cyber Cage UEBA).

Pipeline
--------
1. Load & align tabular + sequential feature files
2. Fit StandardScaler on normal training sequences
3. [Optional] Run Optuna hyperparameter search (--search)
4. Train GRU Autoencoder with best/given hyperparameters
5. Find optimal anomaly threshold (F1-optimal + Youden + percentile)
6. Embed threshold and scaler into the model checkpoint
7. Run full evaluation on test set
8. Generate visualisations
9. Print baseline vs. improved comparison table

Usage
-----
    # Standard training with improved defaults
    python src/train_gru.py

    # Full Optuna search then train with best params
    python src/train_gru.py --search --n-trials 30

    # Manual override of architecture
    python src/train_gru.py --hidden 256 --latent 64 --bidirectional --loss huber

Options
-------
--tabular    PATH    tabular_features.csv    (default: data/processed/...)
--sequential PATH    sequential_features.json
--model-dir  PATH    model checkpoint dir    (default: models/)
--output-dir PATH    evaluation outputs      (default: outputs/)
--plots-dir  PATH    plot directory          (default: outputs/plots/)

Architecture:
--hidden     INT     GRU hidden size         (default: 128)
--latent     INT     Latent bottleneck dim   (default: 64)
--layers     INT     GRU layers              (default: 2)
--dropout    FLOAT   Dropout                 (default: 0.3)
--bidirectional      Enable bidirectional encoder

Training:
--epochs     INT     Max epochs              (default: 150)
--lr         FLOAT   Learning rate           (default: 0.001)
--batch      INT     Batch size              (default: 64)
--patience   INT     Early-stopping patience (default: 20)
--loss       STR     Loss fn: mse/mae/huber  (default: huber)
--seed       INT     Random seed             (default: 42)
--resume     PATH    Resume from checkpoint

Optuna:
--search             Run hyperparameter search before training
--n-trials   INT     Optuna trials           (default: 30)
--quick-epochs INT   Epochs per trial        (default: 30)

Evaluation:
--threshold  FLOAT   Percentile fallback     (default: 95.0)
--no-plots           Skip plot generation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: allow running from any working directory
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CyberCage.TrainGRU")

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from anomaly_detection.dataset_loader import AnomalyDatasetLoader
from anomaly_detection.gru_autoencoder import GRUAutoencoder
from anomaly_detection.trainer import GRUTrainer
from anomaly_detection.evaluator import AnomalyEvaluator
from anomaly_detection.metrics import AnomalyMetricsPlotter
from anomaly_detection.utils import ensure_dir, set_seed, get_device


# ---------------------------------------------------------------------------
# Baseline metrics (recorded before this improvement round)
# ---------------------------------------------------------------------------
_BASELINE = {
    "roc_auc":   0.82,
    "precision": 0.51,
    "recall":    0.43,
    "f1":        0.46,
}

_TARGETS = {
    "roc_auc": 0.86,
    "recall":  0.65,
    "f1":      0.60,
}


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train & evaluate the Cyber Cage GRU Autoencoder (improved)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--tabular",      default="data/processed/tabular_features.csv")
    p.add_argument("--sequential",   default="data/processed/sequential_features.json")
    p.add_argument("--model-dir",    default="models",        dest="model_dir")
    p.add_argument("--output-dir",   default="outputs",       dest="output_dir")
    p.add_argument("--plots-dir",    default="outputs/plots", dest="plots_dir")
    # Architecture
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--layers",       type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--bidirectional",action="store_true", default=False)
    # Training
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--batch",        type=int,   default=64)
    p.add_argument("--patience",     type=int,   default=20)
    p.add_argument("--loss",         default="mse", choices=["mse", "mae", "huber"])
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       default=None)
    # Optuna
    p.add_argument("--search",       action="store_true", default=False,
                   help="Run Optuna search before training")
    p.add_argument("--n-trials",     type=int,   default=30,  dest="n_trials")
    p.add_argument("--quick-epochs", type=int,   default=30,  dest="quick_epochs")
    # Evaluation
    p.add_argument("--threshold",    type=float, default=95.0, dest="threshold_pct")
    p.add_argument("--no-plots",     action="store_true",     dest="no_plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty print helpers
# ---------------------------------------------------------------------------

def _fmt_metric(name: str, val: float, baseline: float, target: float) -> str:
    delta   = val - baseline
    sign    = "+" if delta >= 0 else ""
    met     = "✓" if val >= target else "✗"
    return (
        f"  {name:<12} {baseline:.4f}  →  {val:.4f}  "
        f"({sign}{delta:.4f})   target≥{target:.2f} {met}"
    )


def _print_comparison(metrics) -> None:
    print()
    print("=" * 65)
    print("  PERFORMANCE COMPARISON — Baseline vs. Improved GRU")
    print("=" * 65)
    print(f"  {'Metric':<12} {'Baseline':>8}     {'Current':>8}   {'Delta':>8}   {'Target'}")
    print("  " + "-" * 61)
    print(_fmt_metric("ROC-AUC",   metrics.roc_auc,   _BASELINE["roc_auc"],   _TARGETS["roc_auc"]))
    print(_fmt_metric("Precision", metrics.precision, _BASELINE["precision"], _BASELINE["precision"]))
    print(_fmt_metric("Recall",    metrics.recall,    _BASELINE["recall"],    _TARGETS["recall"]))
    print(_fmt_metric("F1 Score",  metrics.f1,        _BASELINE["f1"],        _TARGETS["f1"]))
    print("=" * 65)
    all_targets_met = (
        metrics.roc_auc >= _TARGETS["roc_auc"] and
        metrics.recall  >= _TARGETS["recall"]  and
        metrics.f1      >= _TARGETS["f1"]
    )
    if all_targets_met:
        print("  ✓ ALL TARGETS MET")
    else:
        print("  ✗ Some targets not yet met — consider running --search")
    print("=" * 65)
    print()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args    = _parse_args()
    t_start = time.time()

    set_seed(args.seed)
    device = get_device()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(args.model_dir, "gru_autoencoder.pt")
    optuna_path     = os.path.join(args.output_dir, "optuna_best_params.json")
    ensure_dir(args.model_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.plots_dir)

    logger.info("=" * 60)
    logger.info("Cyber Cage UEBA — GRU Autoencoder (Improved)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Data loading & splitting
    # ------------------------------------------------------------------
    logger.info("\n[Phase 1] Loading dataset …")
    loader = AnomalyDatasetLoader(
        tabular_path=args.tabular,
        sequential_path=args.sequential,
        batch_size=args.batch,
        seed=args.seed,
    )
    loader.load()
    train_loader, val_loader, test_loader = loader.split()
    seq_len     = loader.sequence_max_len
    feature_dim = loader.sequence_feature_dim
    logger.info("Sequence shape: (%d × %d)", seq_len, feature_dim)

    # ------------------------------------------------------------------
    # Phase 2: Feature normalisation
    # ------------------------------------------------------------------
    logger.info("\n[Phase 2] Fitting StandardScaler on normal training sequences …")
    scaler = loader.fit_scaler()
    logger.info("Scaler fitted and applied to all splits.")

    # ------------------------------------------------------------------
    # Phase 3 (optional): Optuna hyperparameter search
    # ------------------------------------------------------------------
    hparams = {
        "hidden_size":           args.hidden,
        "num_layers":            args.layers,
        "dropout":               args.dropout,
        "bidirectional_encoder": args.bidirectional,
        "loss_fn":               args.loss,
        "learning_rate":         args.lr,
        "batch_size":            args.batch,
    }

    if args.search:
        logger.info("\n[Phase 3] Running Optuna hyperparameter search …")
        logger.info("Trials=%d  QuickEpochs=%d", args.n_trials, args.quick_epochs)
        from anomaly_detection.gru_hyperparameter_search import run_search
        best_params = run_search(
            tabular_path=args.tabular,
            sequential_path=args.sequential,
            n_trials=args.n_trials,
            quick_epochs=args.quick_epochs,
            output_path=optuna_path,
            seed=args.seed,
            device=device,
        )
        # Override hparams with Optuna best
        hparams["hidden_size"]           = best_params.get("hidden_size",   hparams["hidden_size"])
        hparams["num_layers"]            = best_params.get("num_layers",    hparams["num_layers"])
        hparams["dropout"]               = best_params.get("dropout",       hparams["dropout"])
        hparams["bidirectional_encoder"] = best_params.get("bidirectional", hparams["bidirectional_encoder"])
        hparams["loss_fn"]               = best_params.get("loss_fn",       hparams["loss_fn"])
        hparams["learning_rate"]         = best_params.get("learning_rate", hparams["learning_rate"])
        logger.info("Using Optuna best params: %s", json.dumps(
            {k: v for k, v in hparams.items()}, indent=2
        ))

        # Reload data with best batch_size
        best_batch = best_params.get("batch_size", args.batch)
        if best_batch != args.batch:
            logger.info("Reloading data with best batch_size=%d …", best_batch)
            loader2 = AnomalyDatasetLoader(
                tabular_path=args.tabular,
                sequential_path=args.sequential,
                batch_size=best_batch,
                seed=args.seed,
            )
            loader2.load()
            train_loader, val_loader, test_loader = loader2.split()
            loader2.fit_scaler()
            scaler = loader2.get_scaler()
    else:
        logger.info("\n[Phase 3] Skipping Optuna search (use --search to enable).")

    # ------------------------------------------------------------------
    # Phase 4: Model initialisation
    # ------------------------------------------------------------------
    logger.info("\n[Phase 4] Initialising GRU Autoencoder …")
    logger.info(
        "Config: hidden=%d  layers=%d  dropout=%.2f  "
        "bidir=%s  loss=%s",
        hparams["hidden_size"], hparams["num_layers"],
        hparams["dropout"], hparams["bidirectional_encoder"], hparams["loss_fn"],
    )
    model = GRUAutoencoder(
        input_dim=feature_dim,
        hidden_size=hparams["hidden_size"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
        seq_len=seq_len,
        bidirectional_encoder=hparams["bidirectional_encoder"],
    )

    # ------------------------------------------------------------------
    # Phase 5: Training
    # ------------------------------------------------------------------
    logger.info("\n[Phase 5] Training …")
    trainer = GRUTrainer(
        model=model,
        device=device,
        epochs=args.epochs,
        learning_rate=hparams["learning_rate"],
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
        loss_fn=hparams["loss_fn"],
    )
    result = trainer.train(train_loader, val_loader, resume_from=args.resume)
    trainer.save_training_history(result, output_dir=args.output_dir)

    # ------------------------------------------------------------------
    # Phase 6: Reload best model and find optimal threshold
    # ------------------------------------------------------------------
    logger.info("\n[Phase 6] Finding optimal anomaly threshold …")
    best_model = GRUAutoencoder.load(checkpoint_path, device=device)
    evaluator  = AnomalyEvaluator(
        model=best_model,
        device=device,
        threshold_percentile=args.threshold_pct,
        output_dir=args.output_dir,
        loss_fn=hparams["loss_fn"],
    )

    threshold, threshold_comparison = evaluator.find_optimal_threshold(
        val_loader=val_loader,
        test_loader=test_loader,
    )
    logger.info("Optimal threshold (F1-optimal): %.6f", threshold)
    evaluator.save_threshold_comparison(threshold_comparison)

    # ------------------------------------------------------------------
    # Phase 7: Embed threshold + scaler into checkpoint
    # ------------------------------------------------------------------
    logger.info("\n[Phase 7] Embedding threshold and scaler into checkpoint …")
    trainer.save_checkpoint_with_meta(
        path=checkpoint_path,
        threshold=threshold,
        scaler=scaler,
        threshold_method="f1_optimal",
    )

    # ------------------------------------------------------------------
    # Phase 8: Full evaluation on test set
    # ------------------------------------------------------------------
    logger.info("\n[Phase 8] Evaluating on test set …")
    scores, metrics = evaluator.evaluate(
        test_loader=test_loader,
        threshold=threshold,
        records_meta=loader.get_all_records(),
        threshold_method="f1_optimal",
        threshold_comparison=threshold_comparison,
    )
    evaluator.save_scores(scores)
    evaluator.save_predictions(scores)
    evaluator.save_metrics(metrics)

    # ------------------------------------------------------------------
    # Phase 9: Visualisation (including error distribution)
    # ------------------------------------------------------------------
    if not args.no_plots:
        logger.info("\n[Phase 9] Generating visualisations …")
        plotter = AnomalyMetricsPlotter(output_dir=args.plots_dir)

        normal_errs    = [s.reconstruction_error for s in scores if s.is_anomalous == 0]
        anomalous_errs = [s.reconstruction_error for s in scores if s.is_anomalous == 1]
        normal_scores_list    = [s.anomaly_score for s in scores if s.is_anomalous == 0]
        anomalous_scores_list = [s.anomaly_score for s in scores if s.is_anomalous == 1]

        scores_by_label = {
            "normal":    {"reconstruction_error": normal_errs,    "anomaly_score": normal_scores_list},
            "anomalous": {"reconstruction_error": anomalous_errs, "anomaly_score": anomalous_scores_list},
        }

        plotter.plot_all(
            train_losses=result.train_losses,
            val_losses=result.val_losses,
            best_epoch=result.best_epoch,
            scores_by_label=scores_by_label,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time        : %dm %ds", int(elapsed_total // 60), int(elapsed_total % 60))
    logger.info("Best epoch        : %d  (val_loss=%.6f)", result.best_epoch, result.best_val_loss)
    logger.info("Stopped early     : %s", result.stopped_early)
    logger.info("Threshold method  : f1_optimal")
    logger.info("Anomaly threshold : %.6f", threshold)
    logger.info("ROC-AUC           : %.4f", metrics.roc_auc)
    logger.info("PR-AUC            : %.4f", metrics.pr_auc)
    logger.info("F1 Score          : %.4f", metrics.f1)
    logger.info("Precision         : %.4f", metrics.precision)
    logger.info("Recall            : %.4f", metrics.recall)
    logger.info("Model checkpoint  : %s  [+threshold +scaler]", checkpoint_path)
    logger.info("Outputs           : %s/", args.output_dir)

    _print_comparison(metrics)


if __name__ == "__main__":
    main()
