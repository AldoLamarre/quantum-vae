"""Evaluation metrics for Quantum VAE and Classifier trainers."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import numpy as np


def compute_classification_metrics(eval_pred: Union[Tuple[Any, Any], Any]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and top-1 metrics for classification."""
    if isinstance(eval_pred, (tuple, list)):
        logits, labels = eval_pred
    else:
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)

    if logits is None or labels is None:
        return {"accuracy": 0.0}

    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()

    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim > 1:
        preds = np.argmax(logits, axis=-1)
    else:
        preds = (logits > 0.5).astype(int)

    correct = np.sum(preds == labels)
    total = len(labels)
    accuracy = float(correct / max(1, total))

    metrics = {
        "accuracy": accuracy,
        "eval_accuracy": accuracy,
        "num_samples": total,
    }

    # If binary classification, compute precision/recall/f1
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 2:
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return metrics


def compute_vae_metrics(eval_pred: Union[Tuple[Any, Any], Any]) -> Dict[str, float]:
    """Compute MSE reconstruction error and PSNR for VAE evaluation."""
    if isinstance(eval_pred, (tuple, list)):
        reconstructions, targets = eval_pred
    else:
        reconstructions = getattr(eval_pred, "predictions", None)
        targets = getattr(eval_pred, "label_ids", None)

    if reconstructions is None or targets is None:
        return {"reconstruction_mse": 0.0}

    if hasattr(reconstructions, "detach"):
        reconstructions = reconstructions.detach().cpu().numpy()
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy()

    reconstructions = np.asarray(reconstructions)
    targets = np.asarray(targets)

    mse = float(np.mean((reconstructions - targets) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(1e-10, mse))) if mse > 0 else 100.0

    return {
        "reconstruction_mse": mse,
        "psnr": psnr,
    }
