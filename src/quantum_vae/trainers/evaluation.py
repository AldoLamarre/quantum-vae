"""Shared VAE evaluation utilities for trainer and standalone scripts."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class MissingEvalDataError(RuntimeError):
    """Raised when eval_pred has no predictions/targets to compute metrics from.

    This is the one case where a caller may reasonably want to fall back to
    a placeholder result. Any other RuntimeError raised while computing
    metrics (shape mismatches, etc.) is a real bug and should NOT be caught
    the same way.
    """


def normalize_image_range(image_range: str) -> str:
    value = str(image_range).strip().lower()
    if value in {"0_1", "zero_one"}:
        return "0_1"
    if value in {"-1_1", "minus_one_one"}:
        return "-1_1"
    raise ValueError("image_range must be '0_1' or '-1_1'.")


def clamp_to_image_range(tensor: torch.Tensor, image_range: str) -> torch.Tensor:
    normalized = normalize_image_range(image_range)
    if normalized == "0_1":
        return torch.clamp(tensor, 0.0, 1.0)
    return torch.clamp(tensor, -1.0, 1.0)


def to_display_0_1(tensor: torch.Tensor, image_range: str) -> torch.Tensor:
    normalized = normalize_image_range(image_range)
    clamped = clamp_to_image_range(tensor, normalized)
    if normalized == "-1_1":
        clamped = (clamped + 1.0) * 0.5
    return torch.clamp(clamped, 0.0, 1.0)


def extract_input_images(inputs: Any) -> Any:
    if isinstance(inputs, dict):
        for key in ("sample", "pixel_values", "images", "inputs", "x"):
            if key in inputs:
                return inputs[key]
        return next(iter(inputs.values()))
    if isinstance(inputs, (tuple, list)):
        return inputs[0]
    return inputs


def _ensure_4d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(0)
    return tensor


def _ensure_3_channels(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        return tensor
    if tensor.shape[1] == 1:
        return tensor.repeat(1, 3, 1, 1)
    return tensor


def _build_fid_metric(device: torch.device):
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as exc:
        raise RuntimeError(
            "FID evaluation requires torchmetrics FID dependencies. Install 'torch-fidelity'."
        ) from exc

    signature = inspect.signature(FrechetInceptionDistance.__init__)
    kwargs: Dict[str, Any] = {"feature": 2048}
    if "normalize" in signature.parameters:
        kwargs["normalize"] = True
    metric = FrechetInceptionDistance(**kwargs).to(device)
    return metric


def _prepare_eval_pred_tensors(eval_pred: Union[Tuple[Any, Any], Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(eval_pred, (tuple, list)):
        reconstructions, targets = eval_pred
    else:
        reconstructions = getattr(eval_pred, "predictions", None)
        targets = getattr(eval_pred, "label_ids", None)

    if reconstructions is None or targets is None:
        raise MissingEvalDataError("Missing predictions/targets for VAE evaluation metrics.")

    recon_tensor = torch.as_tensor(reconstructions, dtype=torch.float32)
    target_tensor = torch.as_tensor(targets, dtype=torch.float32)
    recon_tensor = _ensure_4d(recon_tensor)
    target_tensor = _ensure_4d(target_tensor)
    return recon_tensor, target_tensor


def compute_reconstruction_metrics_tensors(
    reconstruction: torch.Tensor,
    target_images: torch.Tensor,
    *,
    image_range: str,
) -> Dict[str, float]:
    normalized_range = normalize_image_range(image_range)
    reconstruction = _ensure_4d(reconstruction.float())
    target_images = _ensure_4d(target_images.float())

    recon_clamped = clamp_to_image_range(reconstruction, normalized_range)
    target_clamped = clamp_to_image_range(target_images, normalized_range)

    mse_value = float(F.mse_loss(recon_clamped, target_clamped, reduction="mean").item())
    psnr_value = float(10.0 * torch.log10(torch.tensor(1.0 / max(1e-10, mse_value))).item())

    recon_disp = to_display_0_1(recon_clamped, normalized_range)
    target_disp = to_display_0_1(target_clamped, normalized_range)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).eval()
    ssim_value = float(ssim_metric(recon_disp, target_disp).item())

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg",
        normalize=(normalized_range == "0_1"),
    ).eval()
    recon_lpips = recon_clamped
    target_lpips = target_clamped
    if recon_lpips.shape[1] == 1:
        recon_lpips = _ensure_3_channels(recon_lpips)
        target_lpips = _ensure_3_channels(target_lpips)
    lpips_value = lpips_metric(recon_lpips, target_lpips)
    if hasattr(lpips_value, "mean"):
        lpips_value = lpips_value.mean()

    return {
        "reconstruction_mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "lpips": float(lpips_value.item()),
    }


def compute_reconstruction_metrics_eval_pred(
    eval_pred: Union[Tuple[Any, Any], Any],
    *,
    image_range: str,
) -> Dict[str, float]:
    recon_tensor, target_tensor = _prepare_eval_pred_tensors(eval_pred)
    return compute_reconstruction_metrics_tensors(
        recon_tensor,
        target_tensor,
        image_range=image_range,
    )


class IncrementalVAEMetrics:
    """Stateful `compute_metrics` callable for `TrainingArguments(batch_eval_metrics=True)`.

    With batch_eval_metrics=True, transformers calls
    `compute_metrics(eval_pred, compute_result=is_last_batch)` once per eval
    batch instead of concatenating every batch's predictions/targets into
    one big tensor and calling compute_metrics once at the end. That default
    full-gather behavior is what makes evaluating VAE reconstructions on a
    large validation set (e.g. ImageNet's 50k images) a real GPU/host-memory
    risk -- every reconstructed and target image gets held in memory at
    once. This class instead keeps a running weighted sum across calls and
    only returns the final averaged metrics dict on the last batch,
    so the full eval set is never materialized at once.

    SSIM/LPIPS metric objects (the latter wraps a VGG network) are created
    once and reused across the whole evaluation loop rather than reloaded
    per batch.
    """

    def __init__(self, image_range: str = "0_1"):
        self.image_range = normalize_image_range(image_range)
        self._ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).eval()
        self._lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=(self.image_range == "0_1"),
        ).eval()
        self._reset_running_state()

    def _reset_running_state(self) -> None:
        self._mse_sum = 0.0
        self._ssim_sum = 0.0
        self._lpips_sum = 0.0
        self._count = 0

    def __call__(self, eval_pred, compute_result: bool = True) -> Dict[str, float]:
        try:
            reconstruction, target_images = _prepare_eval_pred_tensors(eval_pred)
        except MissingEvalDataError:
            if not compute_result:
                return {}
            self._reset_running_state()
            return {"reconstruction_mse": 0.0}

        reconstruction = _ensure_4d(reconstruction.float())
        target_images = _ensure_4d(target_images.float())
        recon_clamped = clamp_to_image_range(reconstruction, self.image_range)
        target_clamped = clamp_to_image_range(target_images, self.image_range)

        batch_size = recon_clamped.shape[0]
        mse_value = float(F.mse_loss(recon_clamped, target_clamped, reduction="mean").item())

        recon_disp = to_display_0_1(recon_clamped, self.image_range)
        target_disp = to_display_0_1(target_clamped, self.image_range)
        device = recon_disp.device
        ssim_value = float(self._ssim_metric.to(device)(recon_disp, target_disp).item())

        recon_lpips, target_lpips = recon_clamped, target_clamped
        if recon_lpips.shape[1] == 1:
            recon_lpips = _ensure_3_channels(recon_lpips)
            target_lpips = _ensure_3_channels(target_lpips)
        lpips_value = self._lpips_metric.to(device)(recon_lpips, target_lpips)
        if hasattr(lpips_value, "mean"):
            lpips_value = lpips_value.mean()
        lpips_value = float(lpips_value.item())

        self._mse_sum += mse_value * batch_size
        self._ssim_sum += ssim_value * batch_size
        self._lpips_sum += lpips_value * batch_size
        self._count += batch_size

        if not compute_result:
            return {}

        count = max(1, self._count)
        mse = self._mse_sum / count
        result = {
            "reconstruction_mse": mse,
            "psnr": float(10.0 * torch.log10(torch.tensor(1.0 / max(1e-10, mse))).item()),
            "ssim": self._ssim_sum / count,
            "lpips": self._lpips_sum / count,
        }
        self._reset_running_state()
        return result


def evaluate_vae_reconstruction_dataset(
    *,
    model: Any,
    dataset: Any,
    data_collator: Any,
    image_range: str,
    batch_size: int,
    compute_fid: bool = False,
    sample_posterior: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    normalized_range = normalize_image_range(image_range)
    if dataset is None:
        raise ValueError("dataset must not be None.")
    if model is None:
        raise ValueError("model must not be None.")

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="vgg",
        normalize=(normalized_range == "0_1"),
    ).to(model_device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(model_device).eval()
    fid_metric = _build_fid_metric(model_device) if compute_fid else None

    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    total_count = 0

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = extract_input_images(batch)
            if not isinstance(inputs, torch.Tensor):
                raise RuntimeError("Expected tensor image batch for VAE evaluation.")

            target_images = _ensure_4d(inputs.to(model_device))
            forward_out = model(target_images, sample_posterior=sample_posterior, return_dict=False)
            if isinstance(forward_out, (tuple, list)):
                reconstruction = forward_out[0]
            else:
                reconstruction = getattr(forward_out, "sample", forward_out)
            if not isinstance(reconstruction, torch.Tensor):
                raise RuntimeError("VAE reconstruction output must be a torch.Tensor.")

            reconstruction = _ensure_4d(reconstruction)
            batch_count = reconstruction.shape[0]

            recon_clamped = clamp_to_image_range(reconstruction, normalized_range)
            target_clamped = clamp_to_image_range(target_images, normalized_range)
            mse_value = F.mse_loss(recon_clamped, target_clamped, reduction="mean").item()
            mse_sum += mse_value * batch_count
            psnr_sum += (10.0 * torch.log10(torch.tensor(1.0 / max(1e-10, mse_value))).item()) * batch_count

            recon_disp = to_display_0_1(reconstruction, normalized_range)
            target_disp = to_display_0_1(target_images, normalized_range)
            ssim_sum += float(ssim_metric(recon_disp, target_disp).item()) * batch_count

            recon_lpips = recon_clamped
            target_lpips = target_clamped
            if recon_lpips.ndim == 4 and recon_lpips.shape[1] == 1:
                recon_lpips = _ensure_3_channels(recon_lpips)
                target_lpips = _ensure_3_channels(target_lpips)
            lpips_value = lpips_metric(recon_lpips.float(), target_lpips.float())
            if hasattr(lpips_value, "mean"):
                lpips_value = lpips_value.mean()
            lpips_sum += float(lpips_value.item()) * batch_count

            if fid_metric is not None:
                fid_real = _ensure_3_channels(target_disp.float())
                fid_fake = _ensure_3_channels(recon_disp.float())
                fid_metric.update(fid_real, real=True)
                fid_metric.update(fid_fake, real=False)

            total_count += batch_count
            if max_samples is not None and total_count >= int(max_samples):
                break

    if was_training:
        model.train()

    if total_count == 0:
        raise RuntimeError("No samples were evaluated.")

    metrics: Dict[str, float] = {
        "reconstruction_mse": mse_sum / total_count,
        "psnr": psnr_sum / total_count,
        "ssim": ssim_sum / total_count,
        "lpips": lpips_sum / total_count,
    }

    if fid_metric is not None:
        metrics["fid"] = float(fid_metric.compute().item())

    return metrics
