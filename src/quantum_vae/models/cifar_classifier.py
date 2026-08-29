"""Quantum and Hybrid Classifier models.

Supported dataflow pipelines:
- Standard (softmax=True): inputs -> quantumClassifier (qlayer) -> linear projection -> softmax
- Hybrid (softmax=True): inputs -> quantumClassifier (qlayer) -> mlp -> linear projection -> softmax
- Direct measurement / binary (softmax=False): inputs -> quantumClassifier (qlayer) -> [optional mlp] -> direct measurement outputs (e.g. first qubit for binary)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pennylane as qml
    import torch
    import torch.nn as nn
    has_quantum_deps = True
except ImportError:
    qml = None  # type: ignore
    torch = None  # type: ignore
    nn = object  # type: ignore
    has_quantum_deps = False


@dataclass(frozen=True)
class CifarClassifierConfig:
    n_qubits: int = 7
    n_layers: int = 20
    num_labels: int = 10
    measurement_kind: str = "probability"  # probability | expectation
    measurement_pauli: Optional[str] = None  # X | Y | Z (expectation only)
    postprocessing_mlp_enabled: bool = False
    postprocessing_mlp_hidden_dim: int = 128
    softmax_enabled: bool = True  # When True, enables the linear projection to num_labels + softmax activation. When False, uses measurement outputs directly.


class CifarQuantumClassifier(nn.Module):
    """Quantum classifier architecture.

    Dataflow:
    - Multi-class with softmax (softmax=True):
        inputs -> quantumClassifier (qlayer) -> [optional mlp] -> linear projection -> softmax
    - Direct measurement output / binary (softmax=False):
        inputs -> quantumClassifier (qlayer) -> [optional mlp] -> raw measurement output (or first qubit prediction)
    """

    def __init__(self, config: CifarClassifierConfig):
        super().__init__()
        self.config = config
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

        measurement_dim = (2 ** self.config.n_qubits) if self._is_probability() else self.config.n_qubits
        if self.config.postprocessing_mlp_enabled:
            self.mlp = nn.Sequential(
                nn.Linear(measurement_dim, self.config.postprocessing_mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.config.postprocessing_mlp_hidden_dim, measurement_dim),
                nn.ReLU(),
            )
        else:
            self.mlp = nn.Identity()

        # Backward compatibility alias
        self.postprocessing_mlp = self.mlp

        # The linear projection layer corresponds to the softmax layer to project measurement_dim -> num_labels.
        # When softmax_enabled is True, this projection is instantiated and used.
        # When softmax_enabled is False, no projection is used; quantum measurement outputs are used directly.
        if self.config.softmax_enabled:
            self.classifier = nn.Linear(measurement_dim, self.config.num_labels)
        else:
            self.classifier = nn.Identity()

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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 1. Quantum circuit execution: inputs -> quantumClassifier
        measured = self.qlayer(inputs)

        # 2. Optional MLP processing: -> mlp
        features = self.mlp(measured)

        # 3. Projection & Softmax vs Direct Measurement Output:
        if self.config.softmax_enabled:
            # Linear projection to class logits + softmax
            logits = self.classifier(features)
            return torch.softmax(logits, dim=-1)

        # When softmax is false, measurement outputs are used directly.
        # For binary classification with multi-qubit measurement outputs, extract the first qubit/measurement.
        if self.config.num_labels == 2 and features.dim() > 1 and features.shape[-1] > 1:
            return features[:, 0]

        return features
