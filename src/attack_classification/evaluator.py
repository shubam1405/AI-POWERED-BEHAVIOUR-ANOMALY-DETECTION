"""
evaluator.py – Multiclass evaluation for the Attack Classification Engine.

Computes:
    - Per-class precision, recall, F1-score
    - Macro and weighted averages
    - One-vs-Rest ROC-AUC
    - Confusion matrix
    - Full sklearn classification report
    - Exports metrics to JSON
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

from attack_classification.utils import ensure_dir

logger = logging.getLogger("AttackClassification.Evaluator")


class EvaluationResult:
    """Container for attack classification evaluation metrics."""

    def __init__(self) -> None:
        self.accuracy: float = 0.0
        self.f1_macro: float = 0.0
        self.f1_weighted: float = 0.0
        self.precision_macro: float = 0.0
        self.precision_weighted: float = 0.0
        self.recall_macro: float = 0.0
        self.recall_weighted: float = 0.0
        self.roc_auc_ovr: float = 0.0
        self.confusion_matrix: List[List[int]] = []
        self.classification_report: Dict = {}
        self.per_class_roc: Dict[str, Dict[str, List[float]]] = {}
        self.per_class_pr: Dict[str, Dict[str, List[float]]] = {}
        self.n_total: int = 0
        self.n_classes: int = 0
        self.class_names: List[str] = []

    def as_dict(self) -> Dict:
        return {
            "accuracy":           round(self.accuracy, 4),
            "f1_macro":           round(self.f1_macro, 4),
            "f1_weighted":        round(self.f1_weighted, 4),
            "precision_macro":    round(self.precision_macro, 4),
            "precision_weighted": round(self.precision_weighted, 4),
            "recall_macro":       round(self.recall_macro, 4),
            "recall_weighted":    round(self.recall_weighted, 4),
            "roc_auc_ovr":        round(self.roc_auc_ovr, 4),
            "n_total":            self.n_total,
            "n_classes":          self.n_classes,
            "class_names":        self.class_names,
            "confusion_matrix":   self.confusion_matrix,
            "classification_report": self.classification_report,
        }


class AttackEvaluator:
    """Evaluates a trained XGBoost attack classifier.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained model.
    class_names : list of str
        Ordered class labels.
    output_dir : str
        Where to save evaluation outputs.
    """

    def __init__(
        self,
        model: xgb.XGBClassifier,
        class_names: List[str],
        output_dir: str = "outputs",
    ) -> None:
        self.model = model
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.output_dir = output_dir

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> EvaluationResult:
        """Run full evaluation on the test set.

        Parameters
        ----------
        X_test : ndarray of shape ``(n_samples, n_features)``
        y_test : ndarray of shape ``(n_samples,)``

        Returns
        -------
        EvaluationResult
        """
        logger.info("Evaluating on %d test samples …", len(y_test))

        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        r = EvaluationResult()
        r.n_total = len(y_test)
        r.n_classes = self.n_classes
        r.class_names = self.class_names

        # Core metrics
        r.accuracy           = float(accuracy_score(y_test, y_pred))
        r.f1_macro           = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        r.f1_weighted        = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        r.precision_macro    = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        r.precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        r.recall_macro       = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        r.recall_weighted    = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        r.confusion_matrix = cm.tolist()

        # Classification report
        report = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        r.classification_report = report

        # One-vs-Rest ROC-AUC
        try:
            y_bin = label_binarize(y_test, classes=list(range(self.n_classes)))
            if y_bin.shape[1] == 1:
                # Binary edge case
                y_bin = np.hstack([1 - y_bin, y_bin])
            r.roc_auc_ovr = float(roc_auc_score(
                y_bin, y_prob, average="macro", multi_class="ovr"
            ))
        except ValueError as e:
            logger.warning("ROC-AUC computation failed: %s", e)
            r.roc_auc_ovr = 0.0

        # Per-class ROC and PR curves
        y_bin = label_binarize(y_test, classes=list(range(self.n_classes)))
        for i, cls_name in enumerate(self.class_names):
            if y_bin.shape[1] <= i:
                continue
            # ROC
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                r.per_class_roc[cls_name] = {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                }
            except ValueError:
                pass
            # PR
            try:
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                r.per_class_pr[cls_name] = {
                    "precision": prec.tolist(),
                    "recall":    rec.tolist(),
                }
            except ValueError:
                pass

        logger.info(
            "Evaluation complete  Acc=%.4f  F1_macro=%.4f  F1_weighted=%.4f  ROC-AUC=%.4f",
            r.accuracy, r.f1_macro, r.f1_weighted, r.roc_auc_ovr,
        )

        # Print classification report
        report_str = classification_report(
            y_test, y_pred,
            target_names=self.class_names,
            zero_division=0,
        )
        logger.info("Classification Report:\n%s", report_str)

        return r

    def save_predictions(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        session_ids: np.ndarray,
        output_path: str = "outputs/attack_predictions.csv",
    ) -> str:
        """Generate and save attack predictions to CSV.

        Returns
        -------
        str : Path to saved file.
        """
        ensure_dir(Path(output_path).parent)

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        rows = []
        for i in range(len(y_true)):
            pred_idx = int(y_pred[i])
            true_idx = int(y_true[i])
            confidence = float(y_prob[i, pred_idx])

            # Top 3 predictions
            top_indices = np.argsort(y_prob[i])[::-1][:3]
            top_preds = [
                {"attack": self.class_names[idx], "probability": round(float(y_prob[i, idx]), 4)}
                for idx in top_indices
            ]

            rows.append({
                "session_id":      session_ids[i],
                "true_label":      self.class_names[true_idx],
                "predicted_label": self.class_names[pred_idx],
                "confidence":      round(confidence, 4),
                "correct":         int(pred_idx == true_idx),
                "top_1":           top_preds[0]["attack"],
                "top_1_prob":      top_preds[0]["probability"],
                "top_2":           top_preds[1]["attack"] if len(top_preds) > 1 else "",
                "top_2_prob":      top_preds[1]["probability"] if len(top_preds) > 1 else 0.0,
                "top_3":           top_preds[2]["attack"] if len(top_preds) > 2 else "",
                "top_3_prob":      top_preds[2]["probability"] if len(top_preds) > 2 else 0.0,
            })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info("Attack predictions saved → %s  (%d rows)", output_path, len(df))
        return output_path

    def save_metrics(self, result: EvaluationResult) -> str:
        """Save evaluation metrics to JSON."""
        ensure_dir(self.output_dir)
        path = os.path.join(self.output_dir, "classification_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.as_dict(), f, indent=2)
        logger.info("Classification metrics saved → %s", path)
        return path
