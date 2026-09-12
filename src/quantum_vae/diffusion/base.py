"""Shared abstractions for latent-space diffusion over a frozen QuantumVAE*.

Two concrete variants implement this interface:
- diffusion.euclidean:  plain DDPM on the pre-quantum classical latent
                        (z_flat) -- no group structure, ordinary Gaussian
                        noise and MSE-against-noise training target.
- diffusion.su2_angles: diffusion directly on the per-qubit rotation angles
                        fed into AngleEmbedding -- these parameterize
                        SU(2) elements, so noising/training targets must
                        respect that manifold (tangent-space noise + the
                        SU(2) heat kernel), not flat Euclidean DDPM.

The split point between the two is get_training_target(): what the
denoiser network is trained to regress against. Everything downstream
(the trainer's compute_loss, the generic sampling loop) is written against
this interface and does not need to know which variant it's driving.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import nn


class LatentDenoiserBase(nn.Module, ABC):
    """Predicts a training target (see LatentDiffusionScheduleBase) from a
    noisy latent, the diffusion timestep, and a class label.

    Subclasses provide the actual network architecture; this base class only
    fixes the shared conditioning pattern (timestep + class embeddings) and
    the reserved "unconditional" token used for classifier-free guidance.
    """

    def __init__(self, n_classes: int, hidden: int, n_timesteps: int):
        super().__init__()
        self.n_classes = n_classes
        self.n_timesteps = n_timesteps
        self.unconditional_token = n_classes  # reserved index, never a real label

        self.time_embed = nn.Embedding(n_timesteps, hidden)
        self.class_embed = nn.Embedding(n_classes + 1, hidden)

    @abstractmethod
    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict the training target for this noisy input. Shape of both
        input and output is variant-specific (flat vector for Euclidean,
        per-slot tangent vectors for SU(2) angles)."""
        raise NotImplementedError


class LatentDiffusionScheduleBase(ABC):
    """Defines the forward noising process, the training target the
    denoiser regresses against, and the reverse sampling loop. Concrete
    variants differ in whether these operations are flat-Euclidean or
    manifold-aware (SU(2) heat kernel)."""

    T: int

    @abstractmethod
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: produce x_t from clean data x0 and (variant-
        specific) noise at timestep t."""
        raise NotImplementedError

    @abstractmethod
    def get_training_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """What the denoiser network should regress against for this
        (x0, t, noise) triple. Euclidean DDPM: the raw noise itself.
        SU(2) angle diffusion: the heat-kernel-derived score target."""
        raise NotImplementedError

    @abstractmethod
    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Draw the variant-specific noise used by q_sample (flat Gaussian
        vector for Euclidean; per-slot tangent-space Gaussian for SU(2))."""
        raise NotImplementedError

    @abstractmethod
    def sample(
        self,
        model: LatentDenoiserBase,
        shape: Tuple[int, ...],
        y: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """Full reverse diffusion loop, returning a clean sample in the same
        representation q_sample/get_training_target operate on."""
        raise NotImplementedError
