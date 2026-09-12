"""Generate samples from a trained latent diffusion model:

    python scripts/generate_latent_diffusion.py \
        --config configs/paper/vaequantumhugface_mnist_latent_diffusion.json \
        --checkpoint checkpoints/diffusion/vaequantumhugface_mnist_latent_diffusion/<run>/pytorch_model.bin \
        --digit 3 --n-samples 8

Rebuilds the denoiser + schedule + frozen base VAE the same way
train_latent_diffusion.py does (via TrainerConfigParser), loads trained
denoiser weights, samples, and decodes through the frozen base VAE's real
quantum circuit -- the only place in this script that touches PennyLane.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import TrainerConfigParser
from src.quantum_vae.trainers.latent_diffusion_data import LatentCacheDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained denoiser weights")
    parser.add_argument("--digit", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="generated")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text())

    cfg_parser = TrainerConfigParser()
    parsed = cfg_parser.parse(config)

    model = cfg_parser.build_model(parsed)
    state = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    variant = str(parsed.model_name).lower()
    if variant == "su2_angles":
        from src.quantum_vae.diffusion.su2_angles import SU2HeatKernelSchedule
        schedule = SU2HeatKernelSchedule(
            n_qubits=int(parsed.model_kwargs.get("n_qubits", 10)),
            n_timesteps=int(parsed.model_kwargs.get("n_timesteps", 1000)),
        )
    else:
        from src.quantum_vae.diffusion.euclidean import GaussianDiffusionSchedule
        schedule = GaussianDiffusionSchedule(n_timesteps=int(parsed.model_kwargs.get("n_timesteps", 1000)))

    base_vae = cfg_parser.build_frozen_base_vae(
        parsed.model_kwargs["base_checkpoint"],
        parsed.model_kwargs["base_model_config"],
    )

    cache_path = Path(parsed.data_kwargs["latents_cache"])
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    cache = LatentCacheDataset(str(cache_path))

    latent_shape = tuple(parsed.model_kwargs.get("latent_shape", (4, 7, 7)))
    y = torch.full((args.n_samples,), args.digit, dtype=torch.long)

    with torch.no_grad():
        latent_norm = schedule.sample(model, (args.n_samples, *latent_shape), y, "cpu", guidance_scale=args.guidance_scale)
        latent = latent_norm * cache.std + cache.mean

        z_quantum = base_vae.process_latent(latent)
        images = base_vae.decode(z_quantum).sample

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_samples):
        save_image(torch.clamp(images[i], 0, 1), out_dir / f"digit{args.digit}_sample{i}.png")
    print(f"saved {args.n_samples} samples of digit {args.digit} to {out_dir}")


if __name__ == "__main__":
    main()
