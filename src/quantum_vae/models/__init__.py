"""Quantum VAE model classes."""

from .amplitude_classifier import (
    AmplitudeClassifierPipeline,
    ClassifierPipelineConfig,
    PretrainedAnsatzClassifierPipeline,
)
from .ansatz_vae_base import AnsatzVAEBase

__all__ = [
    "ClassifierPipelineConfig",
    "PretrainedAnsatzClassifierPipeline",
    "AmplitudeClassifierPipeline",
    "AnsatzVAEBase",
]

try:
    from .base import QuantumVAEBase
    from .quantum_vae_amplitude import QuantumVAEAmplitude
    from .quantum_vae_datareupload import QuantumVAEDataReupload

    __all__.extend(
        [
            "QuantumVAEBase",
            "QuantumVAEAmplitude",
            "QuantumVAEDataReupload",
        ]
    )
except Exception:
    pass
