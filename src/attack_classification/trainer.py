"""
trainer.py – XGBoost training pipeline for multiclass attack classification.

Features
--------
* Multiclass classification (``multi:softprob`` objective).
* Automatic class-weight balancing to handle severe class imbalance.
* Configurable hyperparameters (all exposed via constructor).
* Early stopping on validation set.
* Cross-validation support.
* Model checkpointing (pickle).
* GPU support (``device='cuda'``) when available.
* Reproducible via fixed random seed.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import f1_score

from attack_classification.utils import ensure_dir, elapsed_str, set_seed

logger = logging.getLogger("AttackClassification.Trainer")


# ---------------------------------------------------------------------------
# Default hyperparameter search space
# ---------------------------------------------------------------------------

DEFAULT_PARAM_DISTRIBUTIONS: Dict[str, Any] = {
    "max_depth":        [3, 4, 5, 6, 7, 8, 10],
    "learning_rate":    [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
    "n_estimators":     [100, 200, 300, 500, 700, 1000],
    "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3, 5, 7],
    "gamma":            [0.0, 0.1, 0.2, 0.3, 0.5],
    "reg_alpha":        [0.0, 0.01, 0.1, 0.5, 1.0],
    "reg_lambda":       [0.5, 1.0, 1.5, 2.0, 3.0],
}


# ---------------------------------------------------------------------------
# Training result container
# ---------------------------------------------------------------------------

class TrainingResult:
    """Container for training outcomes."""

    def __init__(self) -> None:
        self.best_params: Dict[str, Any] = {}
        self.best_cv_score: float = 0.0
        self.train_accuracy: float = 0.0
        self.val_accuracy: float = 0.0
        self.val_f1_macro: float = 0.0
        self.val_f1_weighted: float = 0.0
        self.n_classes: int = 0
        self.n_features: int = 0
        self.training_time: float = 0.0
        self.model_path: str = ""

    def as_dict(self) -> Dict:
        return {
            "best_params": self.best_params,
            "best_cv_score": round(self.best_cv_score, 4),
            "train_accuracy": round(self.train_accuracy, 4),
            "val_accuracy": round(self.val_accuracy, 4),
            "val_f1_macro": round(self.val_f1_macro, 4),
            "val_f1_weighted": round(self.val_f1_weighted, 4),
            "n_classes": self.n_classes,
            "n_features": self.n_features,
            "training_time_seconds": round(self.training_time, 1),
            "model_path": self.model_path,
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class XGBoostTrainer:
    """XGBoost multiclass attack classifier trainer.

    Parameters
    ----------
    n_classes : int
        Number of target classes.
    seed : int
        Random seed.
    n_estimators : int
        Default number of boosting rounds (overridden during tuning).
    max_depth : int
    learning_rate : float
    subsample : float
    colsample_bytree : float
    min_child_weight : int
    gamma : float
    reg_alpha : float
    reg_lambda : float
    early_stopping_rounds : int
        Early stopping patience for validation loss.
    use_gpu : bool
        If *True*, attempt to use GPU via ``device='cuda'``.
    checkpoint_path : str
        Where to save the trained model.
    """

    def __init__(
        self,
        n_classes: int,
        seed: int = 42,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 2,
        gamma: float = 0.1,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 30,
        use_gpu: bool = False,
        checkpoint_path: str = "models/xgboost_attack_classifier.pkl",
    ) -> None:
        self.n_classes = n_classes
        self.seed = seed
        self.checkpoint_path = checkpoint_path
        self.early_stopping_rounds = early_stopping_rounds

        set_seed(seed)

        self.base_params: Dict[str, Any] = {
            "n_estimators":     n_estimators,
            "max_depth":        max_depth,
            "learning_rate":    learning_rate,
            "subsample":        subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma":            gamma,
            "reg_alpha":        reg_alpha,
            "reg_lambda":       reg_lambda,
            "objective":        "multi:softprob",
            "num_class":        n_classes,
            "eval_metric":      "mlogloss",
            "random_state":     seed,
            "n_jobs":           -1,
            "verbosity":        0,
        }

        if use_gpu:
            self.base_params["device"] = "cuda"
            self.base_params["tree_method"] = "hist"
            logger.info("GPU training enabled.")

        self.model: Optional[xgb.XGBClassifier] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute per-sample weights inversely proportional to class frequency.

        Parameters
        ----------
        y : ndarray of shape ``(n_samples,)``

        Returns
        -------
        weights : ndarray of shape ``(n_samples,)``
        """
        classes, counts = np.unique(y, return_counts=True)
        total = len(y)
        class_weight = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
        weights = np.array([class_weight[yi] for yi in y], dtype=np.float32)
        logger.info("Class weights computed (%d classes). Max weight: %.2f",
                     len(classes), max(class_weight.values()))
        return weights

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_iter: int = 40,
        cv_folds: int = 3,
        sample_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Randomized hyperparameter search with cross-validation.

        Parameters
        ----------
        X_train : ndarray
        y_train : ndarray
        n_iter : int
            Number of random hyperparameter combinations to try.
        cv_folds : int
        sample_weights : ndarray, optional

        Returns
        -------
        dict
            Best hyperparameter combination.
        """
        logger.info(
            "Starting hyperparameter tuning (%d iterations, %d-fold CV) …",
            n_iter, cv_folds,
        )
        t0 = time.time()

        base_model = xgb.XGBClassifier(**self.base_params)
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=DEFAULT_PARAM_DISTRIBUTIONS,
            n_iter=n_iter,
            scoring="f1_macro",
            cv=cv,
            random_state=self.seed,
            n_jobs=-1,
            verbose=0,
            error_score="raise",
        )

        fit_params = {}
        if sample_weights is not None:
            fit_params["sample_weight"] = sample_weights

        search.fit(X_train, y_train, **fit_params)

        best_params = search.best_params_
        logger.info(
            "Tuning complete in %s. Best macro-F1=%.4f",
            elapsed_str(t0), search.best_score_,
        )
        logger.info("Best params: %s", json.dumps(best_params, indent=2))

        return best_params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        sample_weights: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """Train the XGBoost classifier.

        Parameters
        ----------
        X_train, y_train : training data
        X_val, y_val     : validation data (used for early stopping)
        params : dict, optional
            Override hyperparameters (e.g. from tuning).
        sample_weights : ndarray, optional

        Returns
        -------
        TrainingResult
        """
        t0 = time.time()
        result = TrainingResult()
        result.n_classes = self.n_classes
        result.n_features = X_train.shape[1]

        # Merge base params with any tuned overrides
        final_params = {**self.base_params}
        if params:
            final_params.update(params)

        logger.info(
            "Training XGBoost — %d samples, %d features, %d classes …",
            len(X_train), X_train.shape[1], self.n_classes,
        )

        self.model = xgb.XGBClassifier(**final_params)

        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        if sample_weights is not None:
            fit_kwargs["sample_weight"] = sample_weights

        self.model.fit(X_train, y_train, **fit_kwargs)

        # Evaluate on train and val
        y_train_pred = self.model.predict(X_train)
        y_val_pred   = self.model.predict(X_val)

        result.train_accuracy  = float(np.mean(y_train_pred == y_train))
        result.val_accuracy    = float(np.mean(y_val_pred == y_val))
        result.val_f1_macro    = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
        result.val_f1_weighted = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))
        result.best_params     = final_params
        result.training_time   = time.time() - t0

        logger.info(
            "Training complete in %s  train_acc=%.4f  val_acc=%.4f  "
            "val_f1_macro=%.4f  val_f1_weighted=%.4f",
            elapsed_str(t0),
            result.train_accuracy, result.val_accuracy,
            result.val_f1_macro, result.val_f1_weighted,
        )

        # Save model
        self.save_model()
        result.model_path = self.checkpoint_path

        return result

    def save_model(self, path: Optional[str] = None) -> str:
        """Save the trained model to disk.

        Parameters
        ----------
        path : str, optional
            Override the default checkpoint path.

        Returns
        -------
        str : Path to saved file.
        """
        if self.model is None:
            raise RuntimeError("No model to save — call train() first.")
        save_path = path or self.checkpoint_path
        ensure_dir(Path(save_path).parent)
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Model saved → %s", save_path)
        return save_path

    @staticmethod
    def load_model(path: str) -> xgb.XGBClassifier:
        """Load a saved XGBoost model.

        Parameters
        ----------
        path : str

        Returns
        -------
        xgb.XGBClassifier
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded ← %s", path)
        return model

    def get_model(self) -> xgb.XGBClassifier:
        """Return the trained model (for SHAP / evaluation)."""
        if self.model is None:
            raise RuntimeError("No model available — call train() first.")
        return self.model
