"""Classifier pipelines based on VAE-backbone composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


import pennylane as qml
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
    """Shared classifier pipeline with VAE-backbone composition."""

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
        self.wires = np.arange(self.config.n_qubits)
        self.dev = qml.device("default.qubit", wires=self.config.n_qubits)
        self.weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.config.n_layers,
            n_wires=self.config.n_qubits,
        )
        self.qlayer = qml.qnn.TorchLayer(
            self.construct_circuit(),
            weight_shapes={"weights": self.weight_shape},
        )
        self._quantum_torch_device = self._infer_quantum_torch_device()

        self._measurement_input_dim = 2 ** self.config.n_qubits
        self.measurement_projection: Optional[nn.Linear] = None

        measurement_dim = self._measurement_input_dim if self._is_probability() else self.config.n_qubits
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

    def _is_probability(self) -> bool:
        return self.config.measurement_kind.lower() == "probability"

    def _pauli_op(self, wire: int):
        pauli = (self.config.measurement_pauli or "Z").upper()
        if pauli == "X":
            return qml.PauliX(wires=wire)
        if pauli == "Y":
            return qml.PauliY(wires=wire)
        return qml.PauliZ(wires=wire)

    def construct_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit_classifier(inputs, weights):
            qml.QubitStateVector(inputs, wires=self.wires)
            qml.StronglyEntanglingLayers(weights, wires=self.wires)
            if self._is_probability():
                return qml.probs(wires=self.wires)
            return [qml.expval(self._pauli_op(i)) for i in self.wires]

        return circuit_classifier

    def _extract_backbone_features(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.vae_backbone_instance is None:
            return inputs
        if hasattr(self.vae_backbone_instance, "get_latent"):
            return self.vae_backbone_instance.get_latent(inputs, sample_posterior=True)
        return inputs

    def _flatten_real_features(self, features: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(features):
            flat = features.flatten(1)
            return torch.cat([flat.real, flat.imag], dim=1)
        return features.flatten(1).float()

    def _to_measurement_state(self, features: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(features):
            flat_complex = features.flatten(1)
            if flat_complex.shape[1] == self._measurement_input_dim:
                return torch.nn.functional.normalize(flat_complex, dim=1)
        flat_real = self._flatten_real_features(features)
        if flat_real.shape[1] != self._measurement_input_dim:
            if self.measurement_projection is None:
                self.measurement_projection = nn.Linear(flat_real.shape[1], self._measurement_input_dim).to(flat_real.device)
            flat_real = self.measurement_projection(flat_real)
        return torch.nn.functional.normalize(flat_real, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        backbone_features = self._extract_backbone_features(inputs)
        measurement_state = self._to_measurement_state(backbone_features)
        measurement_state = measurement_state.to(self._quantum_torch_device)
        measured = self.qlayer(measurement_state)
        measured = measured.to(self.classifier.weight.device)
        features = self.postprocessing_mlp(measured)
        logits = self.classifier(features)
        return logits


class PretrainedAnsatzClassifierPipeline(_VAEClassifierPipelineBase):
    """Classifier pipeline for pretrained ansatz VAE backbones."""

    def __init__(self, config: ClassifierPipelineConfig, vae_backbone_instance: Optional[Any] = None):
        super().__init__(config=config, vae_backbone_instance=vae_backbone_instance)


class AmplitudeClassifierPipeline(_VAEClassifierPipelineBase):
    """Classifier pipeline for amplitude VAE backbones."""

    def __init__(self, config: ClassifierPipelineConfig, vae_backbone_instance: Optional[Any] = None):
        super().__init__(config=config, vae_backbone_instance=vae_backbone_instance)


__all__ = [
    "ClassifierPipelineConfig",
    "PretrainedAnsatzClassifierPipeline",
    "AmplitudeClassifierPipeline",
]
