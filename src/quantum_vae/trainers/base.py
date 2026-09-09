"""Base Hugging Face Trainer integration for Quantum VAE models."""

from __future__ import annotations

import inspect
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

has_torch = True
has_transformers = True


class BaseHFQuantumTrainer(Trainer):
    """Unified base class for Hugging Face Trainer integration with Quantum models."""

    def __init__(self, *args, **kwargs):
        sig = inspect.signature(Trainer.__init__)
        valid_params = sig.parameters.keys()

        if "tokenizer" in kwargs and "tokenizer" not in valid_params and "processing_class" in valid_params:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        dropped = [k for k in kwargs if k not in valid_params]
        if dropped:
            warnings.warn(
                f"BaseHFQuantumTrainer: dropping kwargs not recognized by the "
                f"installed transformers.Trainer.__init__: {dropped}. This is "
                "usually a typo or a kwarg renamed across transformers versions "
                "-- it will be silently ignored otherwise.",
                stacklevel=2,
            )
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        super().__init__(*args, **filtered_kwargs)
