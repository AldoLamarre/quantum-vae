"""Precompute pre-quantum VAE latents (z_flat) for MNIST using the frozen
data-reupload encoder. This is the training set for Option A diffusion:
plain classical DDPM in the VAE's 196-dim latent space, with everything
downstream of the latent (quantum circuit + decoder) kept completely frozen.

No PennyLane needed to run this script -- only the classical encoder half
of the model is used.

    python scripts/extract_latents_option_a.py [--data-dir DATA_DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import TrainerConfigParser
from src.quantum_vae.utils.model_paths import registered_model_path


def main():
    parser_arg = argparse.ArgumentParser(description=__doc__)
    parser_arg.add_argument("--data-dir", type=str, default=None, help="MNIST root (downloaded if missing); defaults to <project_root>/data")
    args = parser_arg.parse_args()
    data_dir = args.data_dir if args.data_dir is not None else str(ROOT / "data")

    cfg = json.load(open(ROOT / "configs/paper/vaequantumhugface_mnist_pretraining_data_reupload11.json"))
    cfg_parser = TrainerConfigParser()
    parsed = cfg_parser.parse(cfg)
    model = cfg_parser.build_model(parsed)

    dummy = torch.zeros(1, 1, 28, 28)
    model.initialize_projections(dummy)
    sd = torch.load(registered_model_path("mnist_datareupload_compat"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    training_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    loader = DataLoader(training_data, batch_size=256, shuffle=False)

    all_latents = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            posterior = model.encode(x).latent_dist
            z = posterior.mode()          # deterministic, matches sample_posterior=False used elsewhere
            # Kept as (batch, 4, 7, 7) -- the VAE's natural latent shape --
            # rather than flattened to 196. A UNet2D denoiser (see
            # diffusion.euclidean) operates directly on this spatial tensor,
            # the same way Stable Diffusion's own latent diffusion does.
            all_latents.append(z)
            all_labels.append(y)

    latents = torch.cat(all_latents, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print("latents shape:", latents.shape, "mean:", latents.mean().item(), "std:", latents.std().item())

    out_dir = ROOT / "diffusion_data"
    out_dir.mkdir(exist_ok=True)
    torch.save({"latents": latents, "labels": labels}, out_dir / "mnist_latents_option_a.pt")
    print("saved", out_dir / "mnist_latents_option_a.pt")


if __name__ == "__main__":
    main()
