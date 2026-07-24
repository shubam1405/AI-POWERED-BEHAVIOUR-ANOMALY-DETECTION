"""
train_gru.py – End-to-end training and evaluation entry point for the
               GRU Autoencoder Anomaly Detection Engine (Cyber Cage UEBA).

Usage
-----
From the project root:

    python src/train_gru.py [options]

Options
-------
--tabular   PATH    Path to tabular_features.csv
                    (default: data/processed/tabular_features.csv)
--sequential PATH   Path to sequential_features.json
                    (default: data/processed/sequential_features.json)
--model-dir PATH    Directory to save model checkpoint (default: models/)
--output-dir PATH   Directory for evaluation outputs  (default: outputs/)
--plots-dir  PATH   Directory for plots               (default: outputs/plots/)
--epochs     INT    Max training epochs (default: 50)
--hidden     INT    GRU hidden size     (default: 128)
--layers     INT    GRU layers          (default: 2)
--dropout    FLOAT  Dropout probability (default: 0.3)
--lr         FLOAT  Learning rate       (default: 0.001)
--batch      INT    Batch size          (default: 64)
--patience   INT    Early-stopping patience (default: 10)
--threshold  FLOAT  Percentile for threshold estimation (default: 95.0)
--seed       INT    Random seed         (default: 42)
--resume     PATH   Resume from checkpoint
--no-plots          Skip plot generation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup: allow running from any working directory
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Also ensure `src/` is on the path for imports like `from features.xxx`
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), "src")
if os.path.isdir(_SRC_DIR) and _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ---------------------------------------------------------------------------
# Logging setup (must happen before any module imports that log)
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
from anomaly_detection.utils import ensure_dir, set_seed


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train & evaluate the Cyber Cage GRU Autoencoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tabular",    default="data/processed/tabular_features.csv")
    p.add_argument("--sequential", default="data/processed/sequential_features.json")
    p.add_argument("--model-dir",  default="models",         dest="model_dir")
    p.add_argument("--output-dir", default="outputs",        dest="output_dir")
    p.add_argument("--plots-dir",  default="outputs/plots",  dest="plots_dir")
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--hidden",     type=int,   default=128)
    p.add_argument("--layers",     type=int,   default=2)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch",      type=int,   default=64)
    p.add_argument("--patience",   type=int,   default=10)
    p.add_argument("--threshold",  type=float, default=95.0, dest="threshold_pct")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--resume",     default=None)
    p.add_argument("--no-plots",   action="store_true", dest="no_plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    t_start = time.time()

    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(args.model_dir, "gru_autoencoder.pt")
    ensure_dir(args.model_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.plots_dir)

    logger.info("=" * 60)
    logger.info("Cyber Cage UEBA — GRU Autoencoder Training Pipeline")
    logger.info("=" * 60)
    logger.info("Tabular features  : %s", args.tabular)
    logger.info("Sequential features: %s", args.sequential)
    logger.info("Checkpoint target : %s", checkpoint_path)

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
    # Phase 2: Model initialisation
    # ------------------------------------------------------------------
    logger.info("\n[Phase 2] Initialising GRU Autoencoder …")
    model = GRUAutoencoder(
        input_dim=feature_dim,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        seq_len=seq_len,
    )

    # ------------------------------------------------------------------
    # Phase 3: Training
    # ------------------------------------------------------------------
    logger.info("\n[Phase 3] Training …")
    trainer = GRUTrainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.lr,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
    )
    result = trainer.train(train_loader, val_loader, resume_from=args.resume)
    trainer.save_training_history(result, output_dir=args.output_dir)

    # ------------------------------------------------------------------
    # Phase 4: Threshold estimation (reload best model)
    # ------------------------------------------------------------------
    logger.info("\n[Phase 4] Estimating anomaly threshold …")
    import torch
    best_model = GRUAutoencoder.load(checkpoint_path)
    evaluator = AnomalyEvaluator(
        model=best_model,
        threshold_percentile=args.threshold_pct,
        output_dir=args.output_dir,
    )
    threshold = evaluator.estimate_threshold(val_loader)

    # ------------------------------------------------------------------
    # Phase 5: Full evaluation
    # ------------------------------------------------------------------
    logger.info("\n[Phase 5] Evaluating on test set …")
    scores, metrics = evaluator.evaluate(
        test_loader,
        threshold=threshold,
        records_meta=loader.get_all_records(),
    )
    evaluator.save_scores(scores)
    evaluator.save_predictions(scores)
    evaluator.save_metrics(metrics)

    # ------------------------------------------------------------------
    # Phase 6: Visualisation
    # ------------------------------------------------------------------
    if not args.no_plots:
        logger.info("\n[Phase 6] Generating visualisations …")
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
    logger.info("Anomaly threshold : %.6f  (%.0f-th percentile)", threshold, args.threshold_pct)
    logger.info("ROC-AUC           : %.4f", metrics.roc_auc)
    logger.info("PR-AUC            : %.4f", metrics.pr_auc)
    logger.info("F1 Score          : %.4f", metrics.f1)
    logger.info("Precision         : %.4f", metrics.precision)
    logger.info("Recall            : %.4f", metrics.recall)
    logger.info("Model checkpoint  : %s", checkpoint_path)
    logger.info("Outputs           : %s/", args.output_dir)
    logger.info("Plots             : %s/", args.plots_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
