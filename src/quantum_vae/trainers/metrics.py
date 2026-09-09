"""Evaluation metrics for Quantum VAE and Classifier trainers."""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union
import warnings
import numpy as np
from .evaluation import MissingEvalDataError, compute_reconstruction_metrics_eval_pred


def compute_classification_metrics(eval_pred: Union[Tuple[Any, Any], Any]) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and top-1 metrics for classification."""
    if isinstance(eval_pred, (tuple, list)):
        logits, labels = eval_pred
    else:
        logits = getattr(eval_pred, "predictions", None)
        labels = getattr(eval_pred, "label_ids", None)

    if logits is None or labels is None:
        warnings.warn(
            "compute_classification_metrics: predictions or labels are missing; "
            "returning a placeholder accuracy=0.0. This is NOT a real measurement -- "
            "check that eval_dataset/predictions are set up correctly.",
            stacklevel=2,
        )
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


def compute_vae_metrics(
    eval_pred: Union[Tuple[Any, Any], Any],
    *,
    image_range: str = "0_1",
) -> Dict[str, float]:
    """Compute non-FID reconstruction metrics for VAE evaluation."""
    try:
        return compute_reconstruction_metrics_eval_pred(eval_pred, image_range=image_range)
    except MissingEvalDataError as exc:
        warnings.warn(
            f"compute_vae_metrics: no eval data available ({exc}); "
            "returning a placeholder reconstruction_mse=0.0. This is NOT a real "
            "measurement -- check that eval_dataset/predictions are set up correctly.",
            stacklevel=2,
        )
        return {"reconstruction_mse": 0.0}
