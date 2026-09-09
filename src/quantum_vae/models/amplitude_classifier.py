"""Classifier pipelines based on VAE-backbone composition.

Class hierarchy (mirrors the VAE model hierarchy in models/__init__.py:
QuantumVAEBase -> AnsatzVAEBase -> {QuantumVAEDataReupload, QuantumVAENeutralAtom}
QuantumVAEBase -> QuantumVAEAmplitude):

    _VAEClassifierPipelineBase          (config/postprocessing/classifier-head
     |                                   plumbing only -- no circuit assumptions)
     |
     +-- AmplitudeClassifierPipeline     (builds its own fresh StronglyEntanglingLayers
     |                                    circuit; correct for QuantumVAEAmplitude backbones)
     |
     +-- AnsatzClassifierPipelineBase    (reuses -- never rebuilds, never
          |                               reinitializes -- an AnsatzVAEBase
          |                               backbone's own pretrained qlayer)
          |
          +-- DataReuploadClassifierPipeline   (backbone: QuantumVAEDataReupload)
          +-- (NeutralAtomClassifierPipeline lands here once the pulse
               variant exists -- same base class, no new logic needed)

Prior to this refactor, PretrainedAnsatzClassifierPipeline was an empty
subclass that inherited _VAEClassifierPipelineBase's circuit/forward
verbatim -- i.e. it silently built a *fresh, unrelated*
QubitStateVector + StronglyEntanglingLayers circuit and fed it an
already-fully-processed get_latent() output, discarding the ansatz
backbone's own pretrained quantum layer entirely instead of reusing it.
That was a bug, not a design choice; this file fixes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


import pennylane as qml
import torch
import torch.nn as nn
has_quantum_deps = True

from .ansatz_vae_base import AnsatzVAEBase


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


class AnsatzClassifierPipelineBase(_VAEClassifierPipelineBase):
    """Shared logic for classifier pipelines built on an AnsatzVAEBase
    backbone (QuantumVAEDataReupload, QuantumVAENeutralAtom, ...).

    Reuses the backbone's own qlayer object directly -- same tensors, same
    pretrained-from-reconstruction values, still trainable unless the
    caller froze it. Never builds a fresh circuit and never reinitializes
    weights: whatever the VAE pretraining stage learned is exactly what
    this pipeline starts fine-tuning from.

    Whether qlayer / project_to_quantum / project_from_quantum / the
    encoder-decoder are trainable at this stage is decided by whoever
    constructed the backbone instance (see
    hf_classifier_config.build_vae_backbone_instance's
    train_quantum_parts / train_projection_layers / freeze_classical_parts
    flags) -- this class only wires modules together, it does not itself
    change any requires_grad flag.
    """

    def __init__(self, config: ClassifierPipelineConfig, vae_backbone_instance: Optional[Any] = None):
        if not isinstance(vae_backbone_instance, AnsatzVAEBase):
            raise TypeError(
                f"{type(self).__name__} requires an AnsatzVAEBase backbone "
                "(e.g. QuantumVAEDataReupload, QuantumVAENeutralAtom), got "
                f"{type(vae_backbone_instance).__name__ if vae_backbone_instance is not None else None}."
            )
        super().__init__(config=config, vae_backbone_instance=vae_backbone_instance)

    def _build_qlayer(self) -> nn.Module:
        return self.vae_backbone_instance.qlayer   # shared reference, not a copy

    def _measurement_dim(self) -> int:
        return int(self.vae_backbone_instance.n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.vae_backbone_instance.get_pre_quantum_features(inputs, sample_posterior=True)
        measured = self.qlayer(x.to(self._quantum_torch_device))
        measured = measured.to(self.classifier.weight.device)
        features = self.postprocessing_mlp(measured)
        return self.classifier(features)


class DataReuploadClassifierPipeline(AnsatzClassifierPipelineBase):
    """Classifier pipeline for a pretrained QuantumVAEDataReupload backbone."""


# Backward-compatible alias: this class used to be an (incorrectly empty)
# direct subclass of _VAEClassifierPipelineBase under this name.
PretrainedAnsatzClassifierPipeline = DataReuploadClassifierPipeline


__all__ = [
    "ClassifierPipelineConfig",
    "AnsatzClassifierPipelineBase",
    "DataReuploadClassifierPipeline",
    "PretrainedAnsatzClassifierPipeline",
    "AmplitudeClassifierPipeline",
]
