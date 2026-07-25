"""
attack_classification package
==============================
XGBoost-based Intelligent Attack Classification Engine for Cyber Cage UEBA.

Classifies anomalous sessions into specific attack categories using
engineered behavioral features merged with GRU Autoencoder anomaly scores.

Sub-modules
-----------
utils            – seed management, common helpers
dataset_loader   – load tabular + reconstruction CSVs, prepare X/y
feature_merger   – merge & validate feature sets
trainer          – XGBoost training with early stopping, class balancing
evaluator        – multiclass metrics (F1, AUC, confusion matrix)
inference        – single/batch session attack classification
metrics          – feature importance extraction and export
visualization    – dark-themed diagnostic plots
"""

from .trainer import XGBoostTrainer
from .evaluator import AttackEvaluator
from .inference import AttackInferenceEngine

__all__ = [
    "XGBoostTrainer",
    "AttackEvaluator",
    "AttackInferenceEngine",
]
