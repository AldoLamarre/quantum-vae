"""Hugging Face Trainer specialized for latent-space diffusion, trained on
top of a frozen QuantumVAE* base model.

Option-agnostic: compute_loss regresses against schedule.get_training_target(),
not a hardcoded "predict the noise" assumption -- so this same trainer class
drives both diffusion.euclidean (Option A, implemented) and
diffusion.su2_angles (Option B, stub) without modification. Only the
schedule + denoiser pair passed in differs between the two.

Mirrors QuantumVAETrainer's structure and conventions:
- compute_loss(model, inputs, return_outputs=False, **kwargs) signature
- periodic generation previews saved during evaluate(), matching
  _save_reconstruction_preview's naming/behavior in vae_trainer.py
- config-driven construction via TrainerConfigParser (see config_parser.py)

The frozen base VAE (encoder + quantum circuit + decoder) is never touched
by the optimizer -- it is held on the trainer as `self.base_vae`, with all
parameters frozen, and used only to decode generated latents into images
for preview logging. Training itself only ever sees precomputed latent
vectors (see latent_diffusion_data.LatentCacheDataset).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torchvision.utils import make_grid, save_image

from .base import BaseHFQuantumTrainer
from ..diffusion.base import LatentDenoiserBase, LatentDiffusionScheduleBase


class LatentDiffusionTrainer(BaseHFQuantumTrainer):
    """Hugging Face Trainer for diffusion over a frozen VAE's latent
    representation -- option-agnostic over which diffusion.* variant is used."""

    def __init__(
        self,
        model: Optional[LatentDenoiserBase] = None,
        diffusion: Optional[LatentDiffusionScheduleBase] = None,
        args: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_init: Optional[Callable[[], Any]] = None,
        compute_metrics: Optional[Callable[[Any], Dict[str, float]]] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Tuple[Optional[Any], Optional[Any]] = (None, None),
        cfg_dropout_prob: float = 0.1,
        base_vae: Optional[Any] = None,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        latent_shape: Tuple[int, ...] = (4, 7, 7),
        preview_every_n_epochs: int = 10,
        preview_digit: int = 3,
        preview_num_images: int = 8,
        **kwargs,
    ):
        if diffusion is None:
            raise ValueError("LatentDiffusionTrainer requires a `diffusion` schedule (see diffusion.base).")
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            **kwargs,
        )
        self.diffusion = diffusion
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.latent_shape = latent_shape
        self.preview_every_n_epochs = preview_every_n_epochs
        self.preview_digit = preview_digit
        self.preview_num_images = preview_num_images

        # Base VAE is a frozen renderer, not a training target: kept off the
        # model entirely (never registered as an nn.Module attribute) so it
        # can't be silently swept into model.parameters() / the optimizer /
        # checkpoint saving. Frozen explicitly here as a second guarantee.
        self.base_vae = base_vae
        if self.base_vae is not None:
            self.base_vae.eval()
            for p in self.base_vae.parameters():
                p.requires_grad_(False)

    def compute_loss(
        self,
        model: LatentDenoiserBase,
        inputs: Union[Dict[str, Any], Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Diffusion training loss, option-agnostic: regress the model's
        prediction against whatever self.diffusion.get_training_target()
        defines for this variant (raw noise for Euclidean DDPM; the SU(2)
        heat-kernel score target for angle diffusion), with
        classifier-free-guidance label dropout."""
        x0 = inputs["latent"]
        y = inputs["label"]
        device = x0.device

        if hasattr(self.diffusion, "to"):
            self.diffusion.to(device)

        if model.training and self.cfg_dropout_prob > 0:
            uncond_mask = torch.rand(y.shape[0], device=device) < self.cfg_dropout_prob
            y = torch.where(uncond_mask, torch.full_like(y, model.unconditional_token), y)

        t = torch.randint(0, self.diffusion.T, (x0.shape[0],), device=device)
        noise = self.diffusion.sample_noise(x0.shape, device)
        x_noisy = self.diffusion.q_sample(x0, t, noise)
        target = self.diffusion.get_training_target(x0, t, noise)

        prediction = model(x_noisy, t, y)
        loss = torch.mean((prediction - target) ** 2)
        return (loss, {"prediction": prediction}) if return_outputs else loss

    def _save_generation_preview(self, tag: str, epoch: int) -> None:
        """Sample new digits through the trained denoiser, decode them with
        the frozen base VAE (quantum circuit included), and save a grid --
        same naming convention as QuantumVAETrainer._save_reconstruction_preview."""
        if self.base_vae is None:
            return

        output_dir = Path(self.args.output_dir) / "generations"
        output_dir.mkdir(parents=True, exist_ok=True)

        device = next(self.model.parameters()).device
        y = torch.full((self.preview_num_images,), self.preview_digit, dtype=torch.long, device=device)

        self.model.eval()
        with torch.no_grad():
            flat_dim = 1
            for d in self.latent_shape:
                flat_dim *= d
            latent_norm = self.diffusion.sample(
                self.model, (self.preview_num_images, flat_dim), y, device, guidance_scale=2.0
            )
            latent = latent_norm * self.latent_std.to(device) + self.latent_mean.to(device)
            z = latent.reshape(self.preview_num_images, *self.latent_shape)

            self.base_vae.to(device)
            z_quantum = self.base_vae.process_latent(z)
            images = self.base_vae.decode(z_quantum).sample
        self.model.train()

        grid = make_grid(torch.clamp(images, 0, 1), nrow=self.preview_num_images)
        save_image(grid, output_dir / f"epoch-{epoch:04d}-{tag}-digit{self.preview_digit}.png")

    def evaluate(self, *args, **kwargs):
        result = super().evaluate(*args, **kwargs)
        current_epoch = getattr(self.state, "epoch", None)
        if current_epoch is not None and self.preview_every_n_epochs > 0:
            if int(current_epoch) % self.preview_every_n_epochs == 0:
                self._save_generation_preview(tag="periodic", epoch=int(current_epoch))
        return result
