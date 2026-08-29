"""Base Hugging Face Trainer integration for Quantum VAE models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    has_torch = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = None  # type: ignore
    has_torch = False

try:
    import transformers
    from transformers import Trainer, TrainingArguments
    from transformers.trainer_callback import TrainerCallback
    has_transformers = True
except ImportError:
    transformers = None  # type: ignore
    Trainer = object  # type: ignore
    TrainingArguments = None  # type: ignore
    TrainerCallback = None  # type: ignore
    has_transformers = False


class StandaloneHFTrainer:
    """Fallback trainer implementing standard Hugging Face Trainer lifecycle."""

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
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model_init = model_init
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.optimizer, self.lr_scheduler = optimizers

        self.state = {
            "epoch": 0.0,
            "global_step": 0,
            "log_history": [],
            "best_metric": None,
        }

        # Resolve output directory
        self.output_dir = getattr(self.args, "output_dir", "checkpoints/trainer_output")
        if isinstance(self.output_dir, str):
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def compute_loss(self, model: Any, inputs: Dict[str, Any], return_outputs: bool = False, **kwargs) -> Any:
        """Compute training loss. Override in specialized trainers."""
        if hasattr(model, "compute_loss"):
            loss = model.compute_loss(inputs)
            return (loss, None) if return_outputs else loss

        if isinstance(inputs, dict) and "labels" in inputs:
            labels = inputs["labels"]
            feat_inputs = inputs.get("inputs", inputs.get("pixel_values", inputs.get("sample")))
            outputs = model(feat_inputs)
            if nn is not None and torch is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(outputs, labels)
            else:
                loss = 0.0
            return (loss, outputs) if return_outputs else loss

        # Default forward
        if isinstance(inputs, dict):
            outputs = model(**inputs)
        else:
            outputs = model(inputs)

        loss = outputs.get("loss", 0.0) if isinstance(outputs, dict) else outputs
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> Any:
        if self.train_dataset is None:
            raise ValueError("Training requires a train_dataset.")
        if not has_torch:
            return self.train_dataset

        batch_size = getattr(self.args, "per_device_train_batch_size", 32)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def get_eval_dataloader(self, eval_dataset: Optional[Any] = None) -> Any:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Evaluation requires an eval_dataset.")
        if not has_torch:
            return dataset

        batch_size = getattr(self.args, "per_device_eval_batch_size", 32)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def log(self, logs: Dict[str, float]) -> None:
        self.state["log_history"].append(logs)

    def train(self) -> Dict[str, Any]:
        """Execute standard training loop."""
        if not has_torch or self.model is None or self.train_dataset is None:
            return {"training_loss": 0.0, "global_step": 0}

        device = getattr(self.args, "device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        if self.optimizer is None:
            lr = getattr(self.args, "learning_rate", 1e-4)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        epochs = int(getattr(self.args, "num_train_epochs", 1))
        dataloader = self.get_train_dataloader()

        self.model.train()
        total_loss = 0.0
        step = 0

        for epoch in range(epochs):
            self.state["epoch"] = float(epoch + 1)
            for batch in dataloader:
                step += 1
                self.state["global_step"] = step

                if isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                elif hasattr(batch, "to"):
                    batch = batch.to(device)

                self.optimizer.zero_grad()
                loss = self.compute_loss(self.model, batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() if hasattr(loss, "item") else float(loss)

                logging_steps = getattr(self.args, "logging_steps", 25)
                if step % logging_steps == 0:
                    self.log({"loss": total_loss / step, "step": step, "epoch": epoch + 1})

        return {
            "training_loss": total_loss / max(1, step),
            "global_step": step,
            "epoch": epochs,
        }

    def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
        """Execute evaluation loop and compute metrics."""
        if not has_torch or self.model is None:
            return {"eval_loss": 0.0}

        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return {"eval_loss": 0.0}

        dataloader = self.get_eval_dataloader(dataset)
        device = getattr(self.args, "device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        self.model.eval()

        total_eval_loss = 0.0
        steps = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                steps += 1
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
                elif hasattr(batch, "to"):
                    batch = batch.to(device)

                loss, outputs = self.compute_loss(self.model, batch, return_outputs=True)
                total_eval_loss += loss.item() if hasattr(loss, "item") else float(loss)

                if isinstance(batch, dict) and "labels" in batch:
                    all_labels.append(batch["labels"].detach().cpu())
                    if outputs is not None:
                        all_preds.append(outputs.detach().cpu())

        metrics = {"eval_loss": total_eval_loss / max(1, steps)}

        if self.compute_metrics is not None and len(all_preds) > 0 and len(all_labels) > 0:
            preds_cat = torch.cat(all_preds, dim=0).numpy()
            labels_cat = torch.cat(all_labels, dim=0).numpy()
            computed = self.compute_metrics((preds_cat, labels_cat))
            metrics.update(computed)

        self.log(metrics)
        return metrics

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Save model checkpoint to directory."""
        target_dir = output_dir or self.output_dir
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        if has_torch and self.model is not None:
            torch.save(self.model.state_dict(), os.path.join(target_dir, "model.pt"))


import inspect

# Base class inherits from transformers.Trainer if available, otherwise StandaloneHFTrainer
BaseParent = Trainer if has_transformers else StandaloneHFTrainer


class BaseHFQuantumTrainer(BaseParent):
    """Unified base class for Hugging Face Trainer integration with Quantum models."""

    def __init__(self, *args, **kwargs):
        if has_transformers:
            # Inspect Trainer.__init__ to only pass supported arguments
            sig = inspect.signature(Trainer.__init__)
            valid_params = sig.parameters.keys()
            
            # Map tokenizer to processing_class if tokenizer is not accepted but processing_class is
            if "tokenizer" in kwargs and "tokenizer" not in valid_params and "processing_class" in valid_params:
                kwargs["processing_class"] = kwargs.pop("tokenizer")

            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            super().__init__(*args, **filtered_kwargs)
        else:
            super().__init__(*args, **kwargs)
