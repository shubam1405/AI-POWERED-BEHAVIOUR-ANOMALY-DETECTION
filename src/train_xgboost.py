"""
train_xgboost.py – End-to-end training and evaluation for the XGBoost
                    Attack Classification Engine (Cyber Cage UEBA Phase 6).

Usage
-----
From the project root::

    python src/train_xgboost.py [options]

Options
-------
--tabular       PATH   tabular_features.csv          (default: data/processed/tabular_features.csv)
--scores        PATH   reconstruction_scores.csv     (default: outputs/reconstruction_scores.csv)
--model-dir     PATH   Model output directory         (default: models/)
--output-dir    PATH   Evaluation outputs directory   (default: outputs/)
--plots-dir     PATH   Plots directory                (default: outputs/plots/)
--tune-iter     INT    RandomizedSearchCV iterations  (default: 40)
--cv-folds      INT    Cross-validation folds         (default: 3)
--seed          INT    Random seed                    (default: 42)
--no-tune               Skip hyperparameter tuning
--no-plots              Skip plot generation
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("CyberCage.TrainXGBoost")

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from attack_classification.dataset_loader import AttackDatasetLoader
from attack_classification.feature_merger import FeatureMergeValidator
from attack_classification.trainer import XGBoostTrainer
from attack_classification.evaluator import AttackEvaluator
from attack_classification.metrics import FeatureImportanceAnalyzer
from attack_classification.visualization import AttackVisualization
from attack_classification.utils import ensure_dir, set_seed, elapsed_str


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train & evaluate the Cyber Cage XGBoost Attack Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tabular",    default="data/processed/tabular_features.csv")
    p.add_argument("--scores",     default="outputs/reconstruction_scores.csv")
    p.add_argument("--model-dir",  default="models",        dest="model_dir")
    p.add_argument("--output-dir", default="outputs",       dest="output_dir")
    p.add_argument("--plots-dir",  default="outputs/plots", dest="plots_dir")
    p.add_argument("--tune-iter",  type=int, default=40,    dest="tune_iter")
    p.add_argument("--cv-folds",   type=int, default=3,     dest="cv_folds")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--no-tune",    action="store_true",     dest="no_tune")
    p.add_argument("--no-plots",   action="store_true",     dest="no_plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    t_start = time.time()

    set_seed(args.seed)
    ensure_dir(args.model_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.plots_dir)

    checkpoint_path = os.path.join(args.model_dir, "xgboost_attack_classifier.pkl")

    logger.info("=" * 60)
    logger.info("Cyber Cage UEBA — XGBoost Attack Classification Pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Phase 1: Data Loading & Merging
    # ------------------------------------------------------------------
    logger.info("\n[Phase 1] Loading and merging datasets …")
    loader = AttackDatasetLoader(
        tabular_path=args.tabular,
        scores_path=args.scores,
        seed=args.seed,
    )
    loader.load()

    # Validate merged data quality
    validator = FeatureMergeValidator(loader.df, loader.feature_names)
    validator.validate(strict=False)
    summary = validator.summary()
    logger.info("Merge validation: %s", "PASSED ✓" if summary["passed"] else "ISSUES FOUND")

    # Export merged dataset
    loader.save_merged(os.path.join(args.output_dir, "merged_dataset.csv"))

    # ------------------------------------------------------------------
    # Phase 2: Splitting
    # ------------------------------------------------------------------
    logger.info("\n[Phase 2] Stratified train / val / test split …")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), (idx_train, idx_val, idx_test) = loader.split()

    session_ids = loader.get_session_ids()

    logger.info("  Train : %d samples", len(y_train))
    logger.info("  Val   : %d samples", len(y_val))
    logger.info("  Test  : %d samples", len(y_test))

    # ------------------------------------------------------------------
    # Phase 3: Class Balancing
    # ------------------------------------------------------------------
    logger.info("\n[Phase 3] Computing class weights …")
    trainer = XGBoostTrainer(
        n_classes=loader.n_classes,
        seed=args.seed,
        checkpoint_path=checkpoint_path,
    )
    sample_weights = trainer.compute_sample_weights(y_train)

    # ------------------------------------------------------------------
    # Phase 4: Hyperparameter Tuning
    # ------------------------------------------------------------------
    best_params = None
    if not args.no_tune:
        logger.info("\n[Phase 4] Hyperparameter tuning …")
        best_params = trainer.tune_hyperparameters(
            X_train, y_train,
            n_iter=args.tune_iter,
            cv_folds=args.cv_folds,
            sample_weights=sample_weights,
        )
    else:
        logger.info("\n[Phase 4] Hyperparameter tuning SKIPPED (--no-tune).")

    # ------------------------------------------------------------------
    # Phase 5: Training
    # ------------------------------------------------------------------
    logger.info("\n[Phase 5] Training final model …")
    result = trainer.train(
        X_train, y_train,
        X_val, y_val,
        params=best_params,
        sample_weights=sample_weights,
    )

    # Save training result
    import json
    ensure_dir(args.output_dir)
    with open(os.path.join(args.output_dir, "training_result.json"), "w") as f:
        json.dump(result.as_dict(), f, indent=2)

    # ------------------------------------------------------------------
    # Phase 6: Evaluation
    # ------------------------------------------------------------------
    logger.info("\n[Phase 6] Evaluating on test set …")
    model = trainer.get_model()
    evaluator = AttackEvaluator(
        model=model,
        class_names=loader.class_names,
        output_dir=args.output_dir,
    )
    eval_result = evaluator.evaluate(X_test, y_test)
    evaluator.save_metrics(eval_result)
    evaluator.save_predictions(
        X_test, y_test,
        session_ids=session_ids[idx_test],
        output_path=os.path.join(args.output_dir, "attack_predictions.csv"),
    )

    # ------------------------------------------------------------------
    # Phase 7: Feature Importance
    # ------------------------------------------------------------------
    logger.info("\n[Phase 7] Extracting feature importance …")
    fi_analyzer = FeatureImportanceAnalyzer(
        model=model,
        feature_names=loader.feature_names,
        output_dir=args.output_dir,
    )
    importance_df = fi_analyzer.compute()
    fi_analyzer.save(importance_df)
    top20 = fi_analyzer.top_n(20, importance_df)

    # ------------------------------------------------------------------
    # Phase 8: Visualization
    # ------------------------------------------------------------------
    if not args.no_plots:
        logger.info("\n[Phase 8] Generating visualizations …")
        viz = AttackVisualization(output_dir=args.plots_dir)
        viz.plot_all(
            importance_df=importance_df,
            confusion_matrix=eval_result.confusion_matrix,
            class_names=loader.class_names,
            per_class_roc=eval_result.per_class_roc,
            per_class_pr=eval_result.per_class_pr,
            roc_auc=eval_result.roc_auc_ovr,
            classification_report=eval_result.classification_report,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total time         : %s", elapsed_str(t_start))
    logger.info("Classes            : %d", loader.n_classes)
    logger.info("Features           : %d", len(loader.feature_names))
    logger.info("Train accuracy     : %.4f", result.train_accuracy)
    logger.info("Val accuracy       : %.4f", result.val_accuracy)
    logger.info("Val F1 (macro)     : %.4f", result.val_f1_macro)
    logger.info("Val F1 (weighted)  : %.4f", result.val_f1_weighted)
    logger.info("Test accuracy      : %.4f", eval_result.accuracy)
    logger.info("Test F1 (macro)    : %.4f", eval_result.f1_macro)
    logger.info("Test F1 (weighted) : %.4f", eval_result.f1_weighted)
    logger.info("Test ROC-AUC (OvR) : %.4f", eval_result.roc_auc_ovr)
    logger.info("Test Precision (macro)  : %.4f", eval_result.precision_macro)
    logger.info("Test Recall (macro)     : %.4f", eval_result.recall_macro)
    logger.info("Model saved        : %s", checkpoint_path)
    logger.info("Top feature        : %s (gain=%.2f)", top20.iloc[0]["feature"], top20.iloc[0]["gain"])
    logger.info("Outputs            : %s/", args.output_dir)
    logger.info("Plots              : %s/", args.plots_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
