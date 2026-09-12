"""Hugging Face Trainer specialized for latent-space diffusion, trained on
top of a frozen QuantumVAE* base model.

Option-agnostic: compute_loss regresses against schedule.get_training_target(),
not a hardcoded "predict the noise" assumption -- so this same trainer class
drives both diffusion.euclidean (Option A) and diffusion.su2_angles
(Option B) without modification to compute_loss itself. Only the
schedule + denoiser pair, and the generation-preview decode path (which
differs structurally between the two -- see _save_generation_preview),
differ between variants.

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
        variant: str = "euclidean",
        cfg_dropout_prob: float = 0.1,
        base_vae: Optional[Any] = None,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        latent_shape: Tuple[int, ...] = (4, 7, 7),
        n_qubits: int = 10,
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
        self.variant = variant
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.latent_mean = latent_mean
        self.latent_std = latent_std
        self.latent_shape = latent_shape
        self.n_qubits = n_qubits
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
        noise = self.diffusion.sample_noise(x0)
        x_noisy = self.diffusion.q_sample(x0, t, noise)
        target = self.diffusion.get_training_target(x0, t, noise)

        prediction = model(x_noisy, t, y)
        loss = torch.mean((prediction - target) ** 2)
        return (loss, {"prediction": prediction}) if return_outputs else loss

    def prediction_step(
        self,
        model: LatentDenoiserBase,
        inputs: Union[Dict[str, Any], Any],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """HF's default prediction_step calls model(**inputs) directly --
        unpacking the raw batch dict ({"latent", "label"}) straight into
        the model's forward(), which expects (x_noisy, t, y) instead. That
        mismatch is exactly what compute_loss exists to translate for
        training; the same translation has to happen here too, or
        evaluation crashes with a TypeError the moment eval_strategy is
        anything other than "no". Reuses compute_loss's own logic directly
        rather than duplicating it.

        Returns (loss, None, None): there are no per-sample logits/labels
        in the classification-metrics sense for a diffusion denoiser, so
        those two positions are left as None, a supported convention for
        loss-only evaluation.
        """
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)

    def _save_generation_preview(self, tag: str, epoch: int) -> None:
        """Sample new digits through the trained denoiser, decode them with
        the frozen base VAE (quantum circuit included), and save a grid --
        same naming convention as QuantumVAETrainer._save_reconstruction_preview.

        Two different decode paths depending on variant, since Option A and
        Option B produce fundamentally different objects:
        - "euclidean": samples a pre-quantum classical latent -> the normal
          base_vae.process_latent() path (encode->project->qlayer->project
          back), same as a real forward pass would use.
        - "su2_angles": samples quaternions directly -> decomposed back to
          (x,y,z) angles (su2_math.xyz_from_quat) -> fed STRAIGHT into the
          frozen qlayer + project_from_quantum, bypassing process_latent
          entirely, since there's no classical latent to project from here
          -- the diffuser generated the gate angles themselves.
        """
        if self.base_vae is None:
            return

        output_dir = Path(self.args.output_dir) / "generations"
        output_dir.mkdir(parents=True, exist_ok=True)

        device = next(self.model.parameters()).device
        y = torch.full((self.preview_num_images,), self.preview_digit, dtype=torch.long, device=device)

        self.model.eval()
        self.base_vae.to(device)
        with torch.no_grad():
            if self.variant == "su2_angles":
                from ..diffusion.su2_math import xyz_from_quat

                quats = self.diffusion.sample(
                    self.model, (self.preview_num_images, self.n_qubits), y, device, guidance_scale=2.0
                )  # (batch, n_qubits, 4)
                angles = xyz_from_quat(quats)  # (batch, n_qubits, 3)
                quantum_input = angles.reshape(self.preview_num_images, -1)  # (batch, n_qubits*3)

                quantum_output = self.base_vae.qlayer(quantum_input)
                z_quantum_flat = self.base_vae.project_from_quantum(quantum_output)
                z_quantum = z_quantum_flat.reshape(self.preview_num_images, *self.latent_shape)
                images = self.base_vae.decode(z_quantum).sample
            else:
                latent_norm = self.diffusion.sample(
                    self.model, (self.preview_num_images, *self.latent_shape), y, device, guidance_scale=2.0
                )
                latent = latent_norm * self.latent_std.to(device) + self.latent_mean.to(device)
                z_quantum = self.base_vae.process_latent(latent)
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
