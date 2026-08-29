"""Quantum VAE model classes.

This module contains the OOP structure for quantum VAE variants:
- QuantumVAEBase: Abstract base class for all quantum VAE experiments
- QuantumVAEAmplitude: Amplitude encoding strategy
- QuantumVAEDataReupload: Parameterized quantum circuit strategy
"""

from .base import QuantumVAEBase
from .cifar_classifier import CifarClassifierConfig, CifarQuantumClassifier
from .quantum_vae_amplitude import QuantumVAEAmplitude
from .quantum_vae_datareupload import QuantumVAEDataReupload

__all__ = [
    "QuantumVAEBase",
    "CifarClassifierConfig",
    "CifarQuantumClassifier",
    "QuantumVAEAmplitude",
    "QuantumVAEDataReupload",
]
