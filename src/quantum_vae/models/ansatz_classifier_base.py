"""Shared logic for classifier pipelines built on an AnsatzVAEBase backbone.

Mirrors ansatz_vae_base.py (AnsatzVAEBase) on the VAE side.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .ansatz_vae_base import AnsatzVAEBase
from .classifier_base import ClassifierPipelineConfig, _VAEClassifierPipelineBase


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
        # Prefer the qlayer's own measurement_dim when it exposes one (e.g.
        # NeutralAtomPulseLayer, which can be n_atoms or 2**n_atoms
        # depending on measurement_kind). Falls back to n_qubits for
        # variants with no such concept (e.g. the gate-model qlayer, which
        # is always exactly n_qubits expectation values).
        qlayer = self.vae_backbone_instance.qlayer
        if hasattr(qlayer, "measurement_dim"):
            return int(qlayer.measurement_dim)
        return int(self.vae_backbone_instance.n_qubits)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.vae_backbone_instance.get_pre_quantum_features(inputs, sample_posterior=True)
        measured = self.qlayer(x.to(self._quantum_torch_device))
        measured = measured.to(self.classifier.weight.device)
        features = self.postprocessing_mlp(measured)
        return self.classifier(features)
