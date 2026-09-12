"""Option A: DDPM diffusion over the frozen VAE's flat pre-quantum latent
(z_flat). No manifold structure here -- this space is just a continuous
vector space (shaped by the VAE's KL regularization toward something
roughly Gaussian), so standard Euclidean diffusion applies directly, the
same way Stable Diffusion's latent-space diffusion does.

Uses diffusers.DDPMScheduler for the actual noise-schedule math (betas,
alphas_cumprod, add_noise, reverse step) rather than reimplementing it --
this project already depends on diffusers for AutoencoderKL, and the
scheduler math is exactly the well-tested, standard DDPM formulation with
no need for a custom version here.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from diffusers import DDPMScheduler

from .base import LatentDenoiserBase, LatentDiffusionScheduleBase


class FlatLatentDenoiser(LatentDenoiserBase):
    """MLP predicting the noise added to a flat latent vector."""

    def __init__(self, latent_dim: int = 196, n_classes: int = 10, hidden: int = 512, n_timesteps: int = 1000):
        super().__init__(n_classes=n_classes, hidden=hidden, n_timesteps=n_timesteps)
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.time_embed(t) + self.class_embed(y)
        return self.net(torch.cat([x_noisy, cond], dim=-1))


class GaussianDiffusionSchedule(LatentDiffusionScheduleBase):
    """Thin adapter over diffusers.DDPMScheduler implementing the shared
    diffusion.base interface. prediction_type='epsilon' (the default) means
    the training target is just the injected noise itself.

    clip_sample=False: DDPMScheduler defaults to clipping samples to [-1, 1]
    at every reverse step, which is correct for pixel-space diffusion but
    wrong here -- these are latent vectors, not pixel values, and clipping
    them would silently corrupt the distribution the VAE's decoder expects.
    """

    def __init__(self, n_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2):
        self.T = n_timesteps
        self.scheduler = DDPMScheduler(
            num_train_timesteps=n_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            prediction_type="epsilon",
            clip_sample=False,
        )

    def to(self, device: torch.device) -> "GaussianDiffusionSchedule":
        # DDPMScheduler moves its internal tensors lazily based on the
        # dtype/device of tensors passed into add_noise/step -- nothing to
        # move explicitly here, kept for interface symmetry with callers
        # that unconditionally call schedule.to(device).
        return self

    def sample_noise(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(x0, noise, t)

    def get_training_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        assert self.scheduler.config.prediction_type == "epsilon"
        return noise

    @torch.no_grad()
    def sample(
        self,
        model: FlatLatentDenoiser,
        shape: Tuple[int, ...],
        y: torch.Tensor,
        device: torch.device,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        self.scheduler.set_timesteps(self.T, device=device)
        x = torch.randn(shape, device=device)
        uncond_y = torch.full_like(y, model.unconditional_token)

        for t in self.scheduler.timesteps:
            t_batch = t.expand(shape[0]).to(device)
            eps_cond = model(x, t_batch, y)
            if guidance_scale != 1.0:
                eps_uncond = model(x, t_batch, uncond_y)
                eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps_pred = eps_cond
            x = self.scheduler.step(eps_pred, t, x).prev_sample
        return x
