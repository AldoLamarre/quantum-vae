"""Option B: diffusion directly on the per-qubit AngleEmbedding rotation
angles (the 30-dim `quantum_input` in QuantumVAEDataReupload.process_latent),
rather than the frozen VAE's flat classical latent.

NOT YET IMPLEMENTED. Left as a stub, matching the shared
diffusion.base interface, so config_parser.py's model_name routing and the
rest of the pipeline shape are already correct once this is filled in --
only this file needs to change.

Why this needs different math than diffusion.euclidean (do not just copy
GaussianDiffusionSchedule here): each qubit's rotation is applied as
RX(x)*RY(y)*RZ(z) -- a composed element of the Lie group SU(2), not an
unconstrained flat vector. Plain per-angle Gaussian noise does not respect
that manifold (it's the Euler-angle equivalent of gimbal lock: uneven
stretching of the noise across the group, tending to collapse samples
toward degenerate configurations). The correct construction, following
Singh, "Lie Group Diffusion Models for Hardware-Aware Quantum Circuit
Synthesis" (arXiv:2606.29636):
    1. Compose (x, y, z) into one SU(2) element (quaternion) per qubit slot.
    2. Forward-noise via a Gaussian tangent-space (Lie algebra) increment,
       mapped back onto the group with the exponential map -- guarantees
       every step stays exactly on SU(2), unlike naive ambient noising.
    3. Train the denoiser against the SU(2) heat-kernel score target
       (the correct, curvature-aware generalization of the flat Gaussian
       score used in diffusion.euclidean), not raw noise.
    4. At generation time, decompose the resulting quaternion back into an
       (x, y, z) triple for AngleEmbedding -- e.g. via a small differentiable
       fit (a handful of Adam steps minimizing the Frobenius distance to the
       target unitary), rather than a closed-form Euler decomposition, which
       carries branch-cut/gimbal-lock singularities of its own.
"""
from __future__ import annotations

from typing import Tuple

import torch

from .base import LatentDenoiserBase, LatentDiffusionScheduleBase


class AngleSlotDenoiser(LatentDenoiserBase):
    """Per-qubit-slot denoiser predicting a 3D tangent vector per slot,
    conditioned on timestep, class label, and slot index -- shape matches
    the Stanford paper's token-transformer, not yet implemented here."""

    def __init__(self, n_qubits: int = 10, n_classes: int = 10, hidden: int = 512, n_timesteps: int = 1000):
        super().__init__(n_classes=n_classes, hidden=hidden, n_timesteps=n_timesteps)
        self.n_qubits = n_qubits
        raise NotImplementedError(
            "AngleSlotDenoiser is a design stub (see module docstring for the intended "
            "architecture) -- not yet implemented. Use model_name='euclidean' for now."
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SU2HeatKernelSchedule(LatentDiffusionScheduleBase):
    """Tangent-space noising + SU(2) heat-kernel denoising target, per
    qubit slot -- not yet implemented."""

    def __init__(self, n_qubits: int = 10, n_timesteps: int = 1000):
        self.T = n_timesteps
        self.n_qubits = n_qubits
        raise NotImplementedError(
            "SU2HeatKernelSchedule is a design stub (see module docstring for the intended "
            "math) -- not yet implemented. Use model_name='euclidean' for now."
        )

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_training_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sample(
        self,
        model: AngleSlotDenoiser,
        shape: Tuple[int, ...],
        y: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        raise NotImplementedError
