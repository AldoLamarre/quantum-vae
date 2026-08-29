"""Data collators for Quantum VAE and Quantum Classifier trainers."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

try:
    import torch
    has_torch = True
except ImportError:
    torch = None  # type: ignore
    has_torch = False


class VAEDataCollator:
    """Collator for VAE training. Accepts image tensors, tuples (img, label), or dicts."""

    def __init__(self, key: str = "sample"):
        self.key = key

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        if not batch:
            return {}

        first = batch[0]
        if isinstance(first, dict):
            # Already dict format
            if self.key in first:
                return {
                    self.key: torch.stack([item[self.key] for item in batch]) if has_torch and isinstance(first[self.key], torch.Tensor) else [item[self.key] for item in batch]
                }
            if "pixel_values" in first:
                return {
                    self.key: torch.stack([item["pixel_values"] for item in batch]) if has_torch and isinstance(first["pixel_values"], torch.Tensor) else [item["pixel_values"] for item in batch]
                }
            return {k: [item[k] for item in batch] for k in first}

        if isinstance(first, (tuple, list)):
            # Dataset returns (image, label)
            images = [item[0] for item in batch]
            if has_torch and isinstance(images[0], torch.Tensor):
                return {self.key: torch.stack(images)}
            return {self.key: images}

        if has_torch and isinstance(first, torch.Tensor):
            return {self.key: torch.stack(batch)}

        return {self.key: batch}


class ClassifierDataCollator:
    """Collator for Classifier training. Formats inputs into {"inputs": tensor, "labels": tensor}."""

    def __init__(self, input_key: str = "inputs", label_key: str = "labels"):
        self.input_key = input_key
        self.label_key = label_key

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        if not batch:
            return {}

        first = batch[0]
        if isinstance(first, dict):
            res = {}
            for k in first:
                vals = [item[k] for item in batch]
                if has_torch and isinstance(vals[0], torch.Tensor):
                    res[k] = torch.stack(vals)
                else:
                    res[k] = vals
            return res

        if isinstance(first, (tuple, list)):
            inputs = [item[0] for item in batch]
            labels = [item[1] for item in batch]

            if has_torch:
                if isinstance(inputs[0], torch.Tensor):
                    inputs_tensor = torch.stack(inputs)
                else:
                    try:
                        inputs_tensor = torch.tensor(inputs)
                    except Exception:
                        inputs_tensor = inputs

                if isinstance(labels[0], torch.Tensor):
                    labels_tensor = torch.stack(labels)
                else:
                    try:
                        labels_tensor = torch.tensor(labels, dtype=torch.long)
                    except Exception:
                        labels_tensor = labels

                return {self.input_key: inputs_tensor, self.label_key: labels_tensor}

            return {self.input_key: inputs, self.label_key: labels}

        return {self.input_key: batch}
