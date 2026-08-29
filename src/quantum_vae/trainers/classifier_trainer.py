"""Hugging Face Trainer specialized for Quantum and Hybrid Classifier models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    has_torch = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    has_torch = False

from .base import BaseHFQuantumTrainer, StandaloneHFTrainer, has_transformers
from .data_collators import ClassifierDataCollator
from .metrics import compute_classification_metrics


class QuantumClassifierTrainer(BaseHFQuantumTrainer):
    """Hugging Face Trainer for Quantum Classifiers (CifarQuantumClassifier, Hybrid models, etc.).

    Handles:
    - Classifier forward pass:
        * Standard quantum classifier: inputs -> quantumClassifier -> softmax
        * Optional hybrid classifier: inputs -> quantumClassifier -> mlp -> softmax
    - Loss computation: CrossEntropy, BCE with logits, MSE, NLL
    - Classification metrics: accuracy, precision, recall, F1
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        args: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_init: Optional[Callable[[], Any]] = None,
        compute_metrics: Optional[Callable[[Any], Dict[str, float]]] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Tuple[Optional[Any], Optional[Any]] = (None, None),
        loss_fn: Optional[Union[str, Callable]] = "cross_entropy",
        **kwargs,
    ):
        if data_collator is None:
            data_collator = ClassifierDataCollator()
        if compute_metrics is None:
            compute_metrics = compute_classification_metrics

        self.loss_fn_name = loss_fn if isinstance(loss_fn, str) else "custom"
        self.custom_loss_fn = loss_fn if callable(loss_fn) else None

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs,
        )

    def _extract_inputs_and_labels(
        self,
        inputs: Union[Dict[str, Any], Any],
    ) -> Tuple[Any, Any]:
        if isinstance(inputs, dict):
            feat_inputs = inputs.get("inputs", inputs.get("pixel_values", inputs.get("sample", inputs.get("features"))))
            labels = inputs.get("labels", inputs.get("targets", inputs.get("target")))
            return feat_inputs, labels
        if isinstance(inputs, (tuple, list)):
            return inputs[0], inputs[1]
        return inputs, None

    def compute_loss(
        self,
        model: Any,
        inputs: Union[Dict[str, Any], Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Compute classifier loss."""
        feat_inputs, labels = self._extract_inputs_and_labels(inputs)

        # Forward pass
        if hasattr(model, "forward"):
            outputs = model(feat_inputs)
        elif callable(model):
            outputs = model(feat_inputs)
        else:
            outputs = feat_inputs

        if labels is None or not has_torch or not isinstance(outputs, torch.Tensor):
            loss = 0.0
            return (loss, outputs) if return_outputs else loss

        # Loss calculation
        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(outputs, labels)
        elif self.loss_fn_name in ("cross_entropy", "ce"):
            # Check if targets are 1D integer class indices
            if labels.dim() == 1 or (labels.dim() == 2 and labels.size(1) == 1):
                labels_clean = labels.view(-1).long()
                # Check if outputs are already softmax probabilities vs raw logits
                if getattr(getattr(model, "config", None), "softmax_enabled", False):
                    # Softmax already applied; use NLL loss on log(probs)
                    log_probs = torch.log(torch.clamp(outputs, min=1e-9, max=1.0))
                    loss = F.nll_loss(log_probs, labels_clean)
                else:
                    loss = F.cross_entropy(outputs, labels_clean)
            else:
                # One-hot or multi-dim targets
                loss = F.cross_entropy(outputs, labels.float())
        elif self.loss_fn_name in ("bce", "bce_with_logits"):
            loss = F.binary_cross_entropy_with_logits(outputs.view(-1), labels.view(-1).float())
        elif self.loss_fn_name == "mse":
            loss = F.mse_loss(outputs.view(-1), labels.view(-1).float())
        else:
            loss = F.cross_entropy(outputs, labels.long())

        return (loss, outputs) if return_outputs else loss
