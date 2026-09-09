"""Classifier pipeline for amplitude VAE backbones.

Mirrors quantum_vae_amplitude.py (QuantumVAEAmplitude) on the VAE side.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


import pennylane as qml
import torch
import torch.nn as nn

from .classifier_base import ClassifierPipelineConfig, _VAEClassifierPipelineBase


class AmplitudeClassifierPipeline(_VAEClassifierPipelineBase):
    """Classifier pipeline for amplitude VAE backbones.

    Builds its own fresh QubitStateVector + StronglyEntanglingLayers
    circuit and feeds it the backbone's get_latent() output (already
    normalized complex amplitudes) -- correct fit, since amplitude
    backbones do no quantum processing of their own to reuse.
    """

    def _build_qlayer(self) -> nn.Module:
        self.wires = np.arange(self.config.n_qubits)
        self.dev = qml.device("default.qubit", wires=self.config.n_qubits)
        self.weight_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.config.n_layers,
            n_wires=self.config.n_qubits,
        )
        self._measurement_input_dim = 2 ** self.config.n_qubits
        self.measurement_projection: Optional[nn.Linear] = None
        return qml.qnn.TorchLayer(
            self._construct_circuit(),
            weight_shapes={"weights": self.weight_shape},
        )

    def _measurement_dim(self) -> int:
        return self._measurement_input_dim if self._is_probability() else self.config.n_qubits

    def _is_probability(self) -> bool:
        return self.config.measurement_kind.lower() == "probability"

    def _pauli_op(self, wire: int):
        pauli = (self.config.measurement_pauli or "Z").upper()
        if pauli == "X":
            return qml.PauliX(wires=wire)
        if pauli == "Y":
            return qml.PauliY(wires=wire)
        return qml.PauliZ(wires=wire)

    def _construct_circuit(self):
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit_classifier(inputs, weights):
            qml.StatePrep(inputs, wires=self.wires)
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
        return self.classifier(features)


__all__ = ["AmplitudeClassifierPipeline"]
