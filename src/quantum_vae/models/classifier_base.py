"""Shared base for classifier pipelines built on VAE-backbone composition.

Mirrors base.py (QuantumVAEBase) on the VAE side: pure plumbing --
config validation, VAE-backbone reference, postprocessing MLP, classifier
head. Makes NO assumption about what kind of quantum circuit a subclass
uses or how inputs reach it; that's entirely delegated to `_build_qlayer()`
/ `_measurement_dim()` / `forward()`.

See ansatz_classifier_base.py / amplitude_classifier.py /
datareupload_classifier.py for the concrete variants, mirroring
ansatz_vae_base.py / quantum_vae_amplitude.py / quantum_vae_datareupload.py
respectively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
has_quantum_deps = True


BaseTorchModule = nn.Module if has_quantum_deps else object


@dataclass(frozen=True)
class ClassifierPipelineConfig:
    classifier_mode: str = "ansatz"  # ansatz | amplitude
    n_qubits: int = 7
    n_layers: int = 20
    num_labels: int = 10
    measurement_kind: str = "probability"  # probability | expectation
    measurement_pauli: Optional[str] = None  # X | Y | Z (expectation only)
    postprocessing_mlp_enabled: bool = False
    postprocessing_mlp_hidden_dim: int = 128
    logits: bool = True
    softmax_enabled: Optional[bool] = None


class _VAEClassifierPipelineBase(BaseTorchModule):
    """Shared plumbing only: config, VAE-backbone reference, postprocessing
    MLP, and the classifier head. Deliberately makes NO assumption about
    what kind of circuit `qlayer` is or how inputs reach it -- that's
    entirely delegated to `_build_qlayer()` / `_measurement_dim()` /
    `forward()`, which every concrete subclass must supply.
    """

    def __init__(self, config: ClassifierPipelineConfig, vae_backbone_instance: Optional[Any] = None):
        if not has_quantum_deps:
            raise ImportError("pennylane/torch dependencies are required for classifier pipelines.")
        super().__init__()
        logits_enabled = config.logits if config.softmax_enabled is None else bool(config.softmax_enabled)
        if not logits_enabled:
            raise ValueError(
                "classifier.logits=false is not implemented yet; measurement-based classifier outputs are not supported in this code path."
            )
        self.config = config
        self.vae_backbone_instance = vae_backbone_instance

        self.qlayer = self._build_qlayer()
        self._quantum_torch_device = self._infer_quantum_torch_device()

        measurement_dim = self._measurement_dim()
        if self.config.postprocessing_mlp_enabled:
            self.postprocessing_mlp = nn.Sequential(
                nn.Linear(measurement_dim, self.config.postprocessing_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.postprocessing_mlp_hidden_dim, measurement_dim),
                nn.ReLU(),
            )
        else:
            self.postprocessing_mlp = nn.Identity()
        self.classifier = nn.Linear(measurement_dim, self.config.num_labels)

    def _build_qlayer(self) -> nn.Module:
        """Return this pipeline's quantum layer. Amplitude builds a fresh
        one; ansatz pipelines bind the backbone's existing one.
        """
        raise NotImplementedError

    def _measurement_dim(self) -> int:
        """Dimensionality of qlayer's output, i.e. what the classifier
        head's input size should be.
        """
        raise NotImplementedError

    def _infer_quantum_torch_device(self) -> torch.device:
        try:
            return next(self.qlayer.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self.qlayer = self.qlayer.to(self._quantum_torch_device)
        return module

    def set_vae_backbone(self, vae_backbone_instance: Any) -> None:
        self.vae_backbone_instance = vae_backbone_instance

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
