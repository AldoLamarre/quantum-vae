"""Option A: DDPM diffusion over the frozen VAE's spatial pre-quantum latent
(shape [batch, 4, 7, 7]). No manifold structure here -- this space is just
a continuous tensor (shaped by the VAE's KL regularization toward something
roughly Gaussian), so standard Euclidean diffusion applies directly, the
same way Stable Diffusion's own latent diffusion does.

Recycles diffusers' own components rather than reinventing them:
- diffusers.DDPMScheduler for the noise-schedule math (betas, alphas_cumprod,
  add_noise, reverse step)
- diffusers.UNet2DModel for the denoiser network, operating directly on the
  latent's natural (4, 7, 7) spatial shape with built-in timestep/class
  conditioning -- not a from-scratch MLP on a flattened 196-dim vector.
"""
from __future__ import annotations

from typing import Tuple

import torch
from diffusers import DDPMScheduler, UNet2DModel

from .base import LatentDenoiserBase, LatentDiffusionScheduleBase


class LatentUNetDenoiser(LatentDenoiserBase):
    """Thin wrapper around diffusers.UNet2DModel operating on the VAE's
    natural (channels, height, width) latent shape.

    No spatial down/up-sampling stages: at (4, 7, 7), the latent is already
    small enough that a UNet with real downsampling stages hits a classic
    problem -- 7 is odd, so downsampling and upsampling by 2 don't invert
    cleanly, producing mismatched skip-connection shapes. A single
    non-downsampling stage (plain ResNet blocks at full resolution) avoids
    that entirely, which is appropriate at this scale anyway -- there isn't
    much multi-scale structure to gain from downsampling a 7x7 map further.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, int, int] = (4, 7, 7),
        n_classes: int = 10,
        hidden_channels: int = 64,
        n_timesteps: int = 1000,
    ):
        super().__init__(n_classes=n_classes, n_timesteps=n_timesteps)
        channels, height, width = latent_shape
        if height != width:
            raise ValueError(f"LatentUNetDenoiser expects a square latent, got {height}x{width}.")
        self.latent_shape = latent_shape

        self.unet = UNet2DModel(
            sample_size=height,
            in_channels=channels,
            out_channels=channels,
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            block_out_channels=(hidden_channels,),
            layers_per_block=2,
            norm_num_groups=min(8, hidden_channels),
            add_attention=False,
            # +1 reserved slot for the "unconditional" token used by
            # classifier-free guidance (see LatentDenoiserBase).
            num_class_embeds=n_classes + 1,
        )

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.unet(x_noisy, t, class_labels=y).sample


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
        model: LatentUNetDenoiser,
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
