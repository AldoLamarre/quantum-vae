"""Hugging Face Trainer integrations and configurations for Quantum VAE models."""

from .base import BaseHFQuantumTrainer, has_transformers
from .vae_trainer import QuantumVAETrainer
from .classifier_trainer import QuantumClassifierTrainer
from .data_collators import VAEDataCollator, ClassifierDataCollator
from .metrics import compute_classification_metrics, compute_vae_metrics
from .evaluation import (
    compute_reconstruction_metrics_eval_pred,
    evaluate_vae_reconstruction_dataset,
)
from .config_parser import (
    TrainerConfigParser,
    build_trainer_from_config,
    train_from_config,
)

__all__ = [
    "BaseHFQuantumTrainer",
    "has_transformers",
    "QuantumVAETrainer",
    "QuantumClassifierTrainer",
    "VAEDataCollator",
    "ClassifierDataCollator",
    "compute_classification_metrics",
    "compute_vae_metrics",
    "compute_reconstruction_metrics_eval_pred",
    "evaluate_vae_reconstruction_dataset",
    "TrainerConfigParser",
    "build_trainer_from_config",
    "train_from_config",
]
