"""Precompute pre-quantum VAE latents (z_flat) for MNIST using the frozen
data-reupload encoder. This is the training set for Option A diffusion:
plain classical DDPM in the VAE's 196-dim latent space, with everything
downstream of the latent (quantum circuit + decoder) kept completely frozen.

No PennyLane needed to run this script -- only the classical encoder half
of the model is used. MNIST is loaded directly from local idx-ubyte files
rather than torchvision's auto-downloader.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
ROOT = Path("/home/claude/quantum-vae")
sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import TrainerConfigParser
from src.quantum_vae.utils.model_paths import registered_model_path


def load_idx_mnist(images_path: Path, labels_path: Path):
    with open(images_path, "rb") as f:
        _, n, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(n, rows, cols)
    with open(labels_path, "rb") as f:
        struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    x = torch.from_numpy(images).float().unsqueeze(1) / 255.0  # [N, 1, 28, 28]
    y = torch.from_numpy(labels).long()
    return x, y


def main():
    cfg = json.load(open(ROOT / "configs/paper/vaequantumhugface_mnist_pretraining_data_reupload11.json"))
    parser = TrainerConfigParser()
    parsed = parser.parse(cfg)
    model = parser.build_model(parsed)

    dummy = torch.zeros(1, 1, 28, 28)
    model.initialize_projections(dummy)
    sd = torch.load(registered_model_path("mnist_datareupload_compat"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    x_train, y_train = load_idx_mnist(
        Path("/home/claude/mnist_raw/train-images-idx3-ubyte"),
        Path("/home/claude/mnist_raw/train-labels-idx1-ubyte"),
    )
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=256, shuffle=False)

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

    out_dir = Path("/home/claude/diffusion_data")
    out_dir.mkdir(exist_ok=True)
    torch.save({"latents": latents, "labels": labels}, out_dir / "mnist_latents_option_a.pt")
    print("saved", out_dir / "mnist_latents_option_a.pt")


if __name__ == "__main__":
    main()

