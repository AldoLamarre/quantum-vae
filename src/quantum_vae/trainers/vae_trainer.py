"""Hugging Face Trainer specialized for Quantum VAE models."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from .base import BaseHFQuantumTrainer
from .data_collators import VAEDataCollator
from .evaluation import IncrementalVAEMetrics, extract_input_images, normalize_image_range
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
        perceptual_weight: float = 0.0,
        noise_after_epoch: Optional[int] = None,
        noise_std: float = 0.1,
        image_range: str = "0_1",
        save_reconstructions: bool = True,
        reconstruction_every_n_epochs: int = 10,
        reconstruction_num_images: int = 8,
        save_test_reconstructions: bool = True,
        **kwargs,
    ):
        if data_collator is None:
            data_collator = VAEDataCollator()

        self.kl_weight = float(kl_weight)
        self.loss_type = str(loss_type).lower()
        self.perceptual_weight = float(perceptual_weight)
        self.noise_after_epoch = noise_after_epoch
        self.noise_std = float(noise_std)
        self.image_range = self._normalize_image_range(image_range)
        if compute_metrics is None:
            compute_metrics = IncrementalVAEMetrics(self.image_range)
        self.save_reconstructions = bool(save_reconstructions)
        self.reconstruction_every_n_epochs = max(1, int(reconstruction_every_n_epochs))
        self.reconstruction_num_images = max(1, int(reconstruction_num_images))
        self.save_test_reconstructions = bool(save_test_reconstructions)
        self._best_eval_loss: Optional[float] = None
        # One timestamp per trainer instance (i.e. per run), used only to
        # namespace the reconstruction preview PNGs -- two runs against the
        # same config/output_dir previously overwrote each other's
        # epoch-NNNN.png silently, since nothing in that path distinguished
        # runs. Checkpoints are deliberately NOT touched by this: their path
        # needs to stay stable for resume_from_checkpoint to work.
        self._run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.lpips_loss = None
        if self.loss_type == "lpips" or self.perceptual_weight > 0:
            self.lpips_loss = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg",
                normalize=(self.image_range == "0_1"),
            ).eval()
        if self.loss_type == "lpips" and self.perceptual_weight > 0:
            warnings.warn(
                "perceptual_weight > 0 has no effect when loss_type='lpips' "
                "(that path is pure LPIPS, kept for reproducibility). "
                "Use loss_type='l1' or 'mse' with perceptual_weight to combine them.",
                stacklevel=2,
            )

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

    def _extract_input_images(self, inputs: Union[Dict[str, Any], Any]) -> Any:
        return extract_input_images(inputs)

    def _normalize_image_range(self, image_range: str) -> str:
        return normalize_image_range(image_range)

    def _clamp_to_image_range(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.image_range == "0_1":
            return torch.clamp(tensor, 0.0, 1.0)
        return torch.clamp(tensor, -1.0, 1.0)

    def _to_display_0_1(self, tensor: torch.Tensor) -> torch.Tensor:
        clamped = self._clamp_to_image_range(tensor)
        if self.image_range == "-1_1":
            clamped = (clamped + 1.0) * 0.5
        return torch.clamp(clamped, 0.0, 1.0)

    def _prepare_lpips_inputs(
        self,
        reconstruction: torch.Tensor,
        target_images: torch.Tensor,
    ) -> Tuple[LearnedPerceptualImagePatchSimilarity, torch.Tensor, torch.Tensor]:
        if self.lpips_loss is None:
            raise RuntimeError("LPIPS loss is not initialized.")
        metric = self.lpips_loss.to(target_images.device)
        recon_lpips = self._clamp_to_image_range(reconstruction).float()
        target_lpips = self._clamp_to_image_range(target_images).float()
        # torchmetrics' LPIPS is VGG-backed and strictly requires 3-channel
        # (RGB) input -- it raises ValueError on 1-channel (grayscale)
        # tensors rather than handling them. Repeat the single channel
        # into 3 identical channels so grayscale datasets (e.g. MNIST)
        # work with loss_type="lpips" the same way they already work with
        # "mse"/"l1". This does not change what the metric measures for
        # RGB datasets (n_channels == 3 is a no-op repeat). Caveat: VGG's
        # features are calibrated on natural RGB photos, not grayscale
        # digit strokes -- this is a standard, correct way to satisfy the
        # shape requirement, not a claim that it's a perfect perceptual
        # metric for this domain.
        if recon_lpips.shape[1] == 1:
            recon_lpips = recon_lpips.repeat(1, 3, 1, 1)
        if target_lpips.shape[1] == 1:
            target_lpips = target_lpips.repeat(1, 3, 1, 1)
        return metric, recon_lpips, target_lpips

    def _extract_dataset_image(self, item: Any) -> Optional[torch.Tensor]:
        if isinstance(item, torch.Tensor):
            return item
        if isinstance(item, dict):
            value = self._extract_input_images(item)
            return value if isinstance(value, torch.Tensor) else None
        if isinstance(item, (tuple, list)) and item:
            return item[0] if isinstance(item[0], torch.Tensor) else None
        return None

    def _build_preview_batch(self, dataset: Any, num_images: int) -> Optional[torch.Tensor]:
        if dataset is None or not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            return None

        images: List[torch.Tensor] = []
        max_items = min(len(dataset), max(num_images * 4, num_images))
        for idx in range(max_items):
            image = self._extract_dataset_image(dataset[idx])
            if image is None:
                continue
            if image.ndim == 2:
                image = image.unsqueeze(0)
            images.append(image)
            if len(images) >= num_images:
                break
        if not images:
            return None
        return torch.stack(images)

    def _save_reconstruction_preview(self, dataset: Any, split: str, tag: str, epoch: int) -> None:
        if not self.save_reconstructions:
            return
        if self.model is None:
            return
        batch = self._build_preview_batch(dataset, self.reconstruction_num_images)
        if batch is None:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                prepared = self._prepare_inputs({"sample": batch})
                input_images = prepared["sample"]
                forward_out = self.model(input_images, sample_posterior=False, return_dict=False)
                if isinstance(forward_out, (tuple, list)):
                    reconstruction = forward_out[0]
                else:
                    reconstruction = getattr(forward_out, "sample", forward_out)
                if not isinstance(reconstruction, torch.Tensor):
                    return

                target_cpu = input_images.detach().cpu()
                recon_cpu = reconstruction.detach().cpu()
        finally:
            if was_training:
                self.model.train()

        target_disp = self._to_display_0_1(target_cpu)
        recon_disp = self._to_display_0_1(recon_cpu)
        diff_disp = torch.clamp(torch.abs(recon_disp - target_disp), 0.0, 1.0)
        triplet_rows = torch.cat([target_disp, recon_disp, diff_disp], dim=3)
        grid = make_grid(triplet_rows, nrow=1)

        output_dir = Path(self.args.output_dir) / "reconstructions" / split / tag / self._run_timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        save_image(grid, output_dir / f"epoch-{epoch:04d}.png")

    def compute_loss(
        self,
        model: Any,
        inputs: Union[Dict[str, Any], Any],
        return_outputs: bool = False,
        **kwargs,
    ) -> Any:
        """Compute VAE loss: reconstruction_loss + kl_weight * kl_div."""
        input_images = self._extract_input_images(inputs)
        target_images = input_images
        model_input_images = input_images

        if isinstance(input_images, torch.Tensor) and getattr(model, "training", False) and self.noise_after_epoch is not None:
            current_epoch = getattr(self.state, "epoch", None)
            if current_epoch is not None and float(current_epoch) >= float(self.noise_after_epoch):
                model_input_images = input_images + torch.randn_like(input_images) * self.noise_std

        # Initialize projections if required (e.g. DataReupload)
        if hasattr(model, "project_to_quantum") and model.project_to_quantum is None:
            if hasattr(model, "initialize_projections"):
                model.initialize_projections(input_images)

        # Forward pass through VAE
        if hasattr(model, "forward"):
            forward_out = model(model_input_images, sample_posterior=True, return_dict=False)
            if isinstance(forward_out, (tuple, list)):
                reconstruction, kl_div, z_quantum = forward_out[0], forward_out[1], forward_out[2]
            else:
                reconstruction = getattr(forward_out, "sample", forward_out)
                kl_div = torch.tensor(0.0, device=input_images.device) if isinstance(input_images, torch.Tensor) else 0.0
                z_quantum = None
        else:
            reconstruction = input_images
            kl_div = 0.0
            z_quantum = None

        if not isinstance(target_images, torch.Tensor):
            loss = 0.0
        else:
            if not isinstance(reconstruction, torch.Tensor):
                raise RuntimeError("VAE reconstruction output must be a torch.Tensor.")
            reconstruction_tensor = reconstruction

            # Calculate reconstruction loss.
            # Uses the LDM/Stable Diffusion convention: sum over all pixels
            # per sample, then mean over the batch -- i.e. reduction="sum"
            # divided by batch size (torch.sum(x)/N is algebraically the
            # same as (per-sample sum).mean()). This must match _compute_kl's
            # convention or kl_weight silently changes meaning.
            batch_size = reconstruction_tensor.size(0)
            if self.loss_type == "l1":
                recon_loss = F.l1_loss(reconstruction_tensor, target_images, reduction="sum") / batch_size
            elif self.loss_type == "lpips":
                if self.lpips_loss is None:
                    raise RuntimeError("LPIPS loss is not initialized.")
                lpips_metric, recon_lpips, sample_lpips = self._prepare_lpips_inputs(reconstruction_tensor, target_images)
                recon_loss = lpips_metric(recon_lpips, sample_lpips)
                if hasattr(recon_loss, "mean"):
                    recon_loss = recon_loss.mean()
            elif self.loss_type == "mse":
                recon_loss = F.mse_loss(reconstruction_tensor, target_images, reduction="sum") / batch_size
            else:
                raise ValueError(
                    f"Unsupported loss_type '{self.loss_type}'. Supported values: "
                    "'mse', 'l1', 'lpips'."
                )

            # LDM/SD's actual recipe adds LPIPS on top of the pixel loss
            # (rec_loss = pixel_loss + perceptual_weight * lpips), rather than
            # using LPIPS alone. Opt-in via perceptual_weight > 0; leaves the
            # pure loss_type="lpips" path above untouched for reproducibility.
            # Scales don't match LDM exactly (their LPIPS term is elementwise
            # and shares the pixel-sum scale; torchmetrics' LPIPS is already a
            # single batch-mean scalar), so perceptual_weight will need its
            # own tuning rather than reusing LDM's default of 1.0.
            if self.perceptual_weight > 0 and self.loss_type in ("l1", "mse"):
                if self.lpips_loss is None:
                    raise RuntimeError("LPIPS loss is not initialized.")
                lpips_metric, recon_lpips, sample_lpips = self._prepare_lpips_inputs(reconstruction_tensor, target_images)
                perceptual_term = lpips_metric(recon_lpips, sample_lpips)
                if hasattr(perceptual_term, "mean"):
                    perceptual_term = perceptual_term.mean()
                recon_loss = recon_loss + self.perceptual_weight * perceptual_term

            loss = recon_loss + self.kl_weight * kl_div

        outputs = {
            "loss": loss,
            "reconstruction": reconstruction,
            "kl_div": kl_div,
            "z_quantum": z_quantum,
        }

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: Any,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        detached_loss: Optional[torch.Tensor]
        if isinstance(loss, torch.Tensor):
            detached_loss = loss.mean().detach()
        else:
            detached_loss = None

        if prediction_loss_only:
            return detached_loss, None, None

        reconstruction = outputs.get("reconstruction") if isinstance(outputs, dict) else None
        target_images = self._extract_input_images(inputs)

        pred_tensor = reconstruction.detach() if isinstance(reconstruction, torch.Tensor) else None
        target_tensor = target_images.detach() if isinstance(target_images, torch.Tensor) else None
        return detached_loss, pred_tensor, target_tensor

    def evaluate(
        self,
        eval_dataset: Optional[Any] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        dataset_for_preview = eval_dataset if eval_dataset is not None else self.eval_dataset
        current_epoch = int(round(float(getattr(self.state, "epoch", 0.0) or 0.0)))
        if current_epoch <= 0:
            current_epoch = 1

        if metric_key_prefix == "eval":
            should_save_periodic = current_epoch == 1 or (current_epoch % self.reconstruction_every_n_epochs == 0)
            if should_save_periodic:
                self._save_reconstruction_preview(dataset_for_preview, split="validation", tag="periodic", epoch=current_epoch)

            eval_loss = metrics.get("eval_loss")
            if isinstance(eval_loss, (float, int)):
                eval_loss_value = float(eval_loss)
                if self._best_eval_loss is None or eval_loss_value < self._best_eval_loss:
                    self._best_eval_loss = eval_loss_value
                    self._save_reconstruction_preview(dataset_for_preview, split="validation", tag="best", epoch=current_epoch)
        elif metric_key_prefix == "test" and self.save_test_reconstructions:
            self._save_reconstruction_preview(dataset_for_preview, split="test", tag="final", epoch=current_epoch)

        return metrics
