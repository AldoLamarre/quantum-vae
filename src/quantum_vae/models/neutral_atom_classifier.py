"""Classifier pipeline for a pretrained QuantumVAENeutralAtom backbone.

Mirrors datareupload_classifier.py on the VAE side. No logic of its own:
AnsatzClassifierPipelineBase treats the backbone's qlayer opaquely via
vae_backbone_instance.qlayer / .n_qubits / .get_pre_quantum_features,
none of which care what circuit family qlayer actually is (gate-model
data re-upload vs. Rydberg pulse program).
"""

from __future__ import annotations

from .ansatz_classifier_base import AnsatzClassifierPipelineBase


class NeutralAtomClassifierPipeline(AnsatzClassifierPipelineBase):
    """Classifier pipeline for a pretrained QuantumVAENeutralAtom backbone."""


__all__ = ["NeutralAtomClassifierPipeline"]
