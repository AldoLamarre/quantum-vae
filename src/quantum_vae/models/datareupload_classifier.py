"""Classifier pipeline for a pretrained QuantumVAEDataReupload backbone.

Mirrors quantum_vae_datareupload.py (QuantumVAEDataReupload) on the VAE
side. No logic of its own: everything variant-specific already lives in
AnsatzClassifierPipelineBase, which treats the backbone's qlayer opaquely
regardless of circuit family -- same reason AmplitudeClassifierPipeline
needs no ansatz-side counterpart logic either.
"""

from __future__ import annotations

from .ansatz_classifier_base import AnsatzClassifierPipelineBase


class DataReuploadClassifierPipeline(AnsatzClassifierPipelineBase):
    """Classifier pipeline for a pretrained QuantumVAEDataReupload backbone."""


# Backward-compatible alias: this class used to be an (incorrectly empty)
# direct subclass of _VAEClassifierPipelineBase under this name, living in
# amplitude_classifier.py.
PretrainedAnsatzClassifierPipeline = DataReuploadClassifierPipeline


__all__ = [
    "DataReuploadClassifierPipeline",
    "PretrainedAnsatzClassifierPipeline",
]
