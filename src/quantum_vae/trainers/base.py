"""Base Hugging Face Trainer integration for Quantum VAE models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer

has_torch = True
has_transformers = True


import inspect

class BaseHFQuantumTrainer(Trainer):
    """Unified base class for Hugging Face Trainer integration with Quantum models."""

    def __init__(self, *args, **kwargs):
        sig = inspect.signature(Trainer.__init__)
        valid_params = sig.parameters.keys()

        if "tokenizer" in kwargs and "tokenizer" not in valid_params and "processing_class" in valid_params:
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        super().__init__(*args, **filtered_kwargs)
