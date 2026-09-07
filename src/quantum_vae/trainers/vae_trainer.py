"""Hugging Face Trainer specialized for Quantum VAE models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .base import BaseHFQuantumTrainer
from .data_collators import VAEDataCollator
from .metrics import compute_vae_metrics
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class QuantumVAETrainer(BaseHFQuantumTrainer):
    """Hugging Face Trainer for Quantum VAE variants (Amplitude, DataReupload, etc.).

    Handles:
    - VAE forward pass (encode -> sample -> quantum encode -> decode)
    - Combined loss: reconstruction_loss + kl_weight * kl_divergence
    - Latent logging & reconstruction metrics
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        args: Optional[Any] = None,
        data_collator: Optional[Callable] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        model_init: Optional[Callable[[], Any]] = None,
        compute_metrics: Optional[Callable[[Any], Dict[str, float]]] = None,
        callbacks: Optional[List[Any]] = None,
        optimizers: Tuple[Optional[Any], Optional[Any]] = (None, None),
        kl_weight: float = 1e-4,
        loss_type: str = "mse",
        noise_after_epoch: Optional[int] = None,
        noise_std: float = 0.1,
        **kwargs,
    ):
        if data_collator is None:
            data_collator = VAEDataCollator()
        if compute_metrics is None:
            compute_metrics = compute_vae_metrics

        self.kl_weight = float(kl_weight)
        self.loss_type = str(loss_type).lower()
        self.noise_after_epoch = noise_after_epoch
        self.noise_std = float(noise_std)
        self.lpips_loss_01 = None
        self.lpips_loss_11 = None
        if self.loss_type == "lpips":
            self.lpips_loss_01 = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).eval()
            self.lpips_loss_11 = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).eval()

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

    def _extract_sample(self, inputs: Union[Dict[str, Any], Any]) -> Any:
        if isinstance(inputs, dict):
            for key in ("sample", "pixel_values", "images", "inputs", "x"):
                if key in inputs:
                    return inputs[key]
            # fallback to first value
            return next(iter(inputs.values()))
        return inputs

    def _prepare_lpips_inputs(
        self,
        reconstruction: torch.Tensor,
        sample: torch.Tensor,
    ) -> Tuple[LearnedPerceptualImagePatchSimilarity, torch.Tensor, torch.Tensor]:
        sample_min = float(sample.detach().amin().item())
        sample_max = float(sample.detach().amax().item())

        if sample_min >= -1e-3 and sample_max <= 1.0 + 1e-3:
            if self.lpips_loss_01 is None:
                raise RuntimeError("LPIPS [0,1] loss is not initialized.")
            metric = self.lpips_loss_01.to(sample.device)
            recon_lpips = torch.clamp(reconstruction, 0.0, 1.0).float()
            sample_lpips = torch.clamp(sample, 0.0, 1.0).float()
            return metric, recon_lpips, sample_lpips

        if self.lpips_loss_11 is None:
            raise RuntimeError("LPIPS [-1,1] loss is not initialized.")
        metric = self.lpips_loss_11.to(sample.device)
        recon_lpips = torch.clamp(reconstruction, -1.0, 1.0).float()
        sample_lpips = torch.clamp(sample, -1.0, 1.0).float()
        return metric, recon_lpips, sample_lpips

    def compute_loss(
        self,
        model: Any,
        inputs: Union[Dict[str, Any], Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Compute VAE loss: reconstruction_loss + kl_weight * kl_div."""
        sample = self._extract_sample(inputs)
        noisy_sample = sample

        if isinstance(sample, torch.Tensor) and getattr(model, "training", False) and self.noise_after_epoch is not None:
            current_epoch = getattr(self.state, "epoch", None)
            if current_epoch is not None and float(current_epoch) >= float(self.noise_after_epoch):
                noisy_sample = sample + torch.randn_like(sample) * self.noise_std

        # Initialize projections if required (e.g. DataReupload)
        if hasattr(model, "project_to_quantum") and model.project_to_quantum is None:
            if hasattr(model, "initialize_projections"):
                model.initialize_projections(sample)

        # Forward pass through VAE
        if hasattr(model, "forward"):
            forward_out = model(noisy_sample, sample_posterior=True, return_dict=False)
            if isinstance(forward_out, (tuple, list)):
                reconstruction, kl_div, z_quantum = forward_out[0], forward_out[1], forward_out[2]
            else:
                reconstruction = getattr(forward_out, "sample", forward_out)
                kl_div = torch.tensor(0.0, device=sample.device) if isinstance(sample, torch.Tensor) else 0.0
                z_quantum = None
        else:
            reconstruction = sample
            kl_div = 0.0
            z_quantum = None

        if not isinstance(sample, torch.Tensor):
            loss = 0.0
        else:
            if not isinstance(reconstruction, torch.Tensor):
                raise RuntimeError("VAE reconstruction output must be a torch.Tensor.")
            reconstruction_tensor = reconstruction

            # Calculate reconstruction loss
            if self.loss_type == "l1":
                recon_loss = F.l1_loss(reconstruction_tensor, sample)
            elif self.loss_type == "bce":
                recon_loss = F.binary_cross_entropy(torch.clamp(reconstruction_tensor, 0.0, 1.0), sample)
            elif self.loss_type == "lpips":
                if self.lpips_loss_01 is None or self.lpips_loss_11 is None:
                    raise RuntimeError("LPIPS loss is not initialized.")
                lpips_metric, recon_lpips, sample_lpips = self._prepare_lpips_inputs(reconstruction_tensor, sample)
                recon_loss = lpips_metric(recon_lpips, sample_lpips)
                if hasattr(recon_loss, "mean"):
                    recon_loss = recon_loss.mean()
            else:
                recon_loss = F.mse_loss(reconstruction_tensor, sample)

            loss = recon_loss + self.kl_weight * kl_div

        outputs = {
            "loss": loss,
            "reconstruction": reconstruction,
            "kl_div": kl_div,
            "z_quantum": z_quantum,
        }

        return (loss, outputs) if return_outputs else loss
