"""
anomaly_detection package
=========================
GRU Autoencoder-based Adaptive Anomaly Detection Engine for Cyber Cage UEBA.

Sub-modules
-----------
utils           – seed management, device selection, common helpers
dataset_loader  – data loading, splitting, PyTorch Dataset/DataLoader
gru_autoencoder – PyTorch GRU Encoder-Decoder model
trainer         – training loop, early stopping, checkpointing
evaluator       – threshold estimation, metrics computation
inference       – single-session scoring
metrics         – plots (loss curves, ROC, histograms)
"""

from .gru_autoencoder import GRUAutoencoder
from .dataset_loader import AnomalyDatasetLoader
from .trainer import GRUTrainer
from .evaluator import AnomalyEvaluator
from .inference import InferenceEngine

__all__ = [
    "GRUAutoencoder",
    "AnomalyDatasetLoader",
    "GRUTrainer",
    "AnomalyEvaluator",
    "InferenceEngine",
]
