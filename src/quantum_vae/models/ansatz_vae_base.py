from __future__ import annotations

from abc import abstractmethod
from typing import Iterable


import torch


from .base import QuantumVAEBase


class AnsatzVAEBase(QuantumVAEBase):
    """Base class for ansatz-style quantum VAE variants."""

    @abstractmethod
    def construct_circuit(self):
        raise NotImplementedError("Subclass must implement construct_circuit().")

    def quantum_trainable_parameters(self) -> Iterable["torch.nn.Parameter"]:
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "qlayer" in name or "project_to_quantum" in name or "project_from_quantum" in name:
                yield param
