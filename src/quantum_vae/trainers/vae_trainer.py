"""Hugging Face Trainer specialized for Quantum VAE models."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    has_torch = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    has_torch = False

from .base import BaseHFQuantumTrainer, StandaloneHFTrainer, has_transformers
from .data_collators import VAEDataCollator
from .metrics import compute_vae_metrics

try:
    from taming.modules.losses.lpips import LPIPS

    has_lpips = True
except ImportError:
    LPIPS = None  # type: ignore
    has_lpips = False


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
        self.lpips_loss = None
        if self.loss_type == "lpips":
            if not has_lpips:
                raise ImportError("LPIPS loss requested but taming.modules.losses.lpips is unavailable.")
            self.lpips_loss = LPIPS().eval()

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

        if (
            has_torch
            and isinstance(sample, torch.Tensor)
            and getattr(model, "training", False)
            and self.noise_after_epoch is not None
        ):
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
                kl_div = torch.tensor(0.0, device=sample.device) if has_torch else 0.0
                z_quantum = None
        else:
            reconstruction = sample
            kl_div = 0.0
            z_quantum = None

        if not has_torch or not isinstance(sample, torch.Tensor):
            loss = 0.0
        else:
            # Calculate reconstruction loss
            if self.loss_type == "l1":
                recon_loss = F.l1_loss(reconstruction, sample)
            elif self.loss_type == "bce":
                recon_loss = F.binary_cross_entropy(torch.clamp(reconstruction, 0.0, 1.0), sample)
            elif self.loss_type == "lpips":
                if self.lpips_loss is None:
                    raise RuntimeError("LPIPS loss is not initialized.")
                self.lpips_loss = self.lpips_loss.to(sample.device)
                recon_loss = self.lpips_loss(reconstruction, sample).mean()
            else:
                recon_loss = F.mse_loss(reconstruction, sample)

            loss = recon_loss + self.kl_weight * kl_div

        outputs = {
            "loss": loss,
            "reconstruction": reconstruction,
            "kl_div": kl_div,
            "z_quantum": z_quantum,
        }

        return (loss, outputs) if return_outputs else loss
