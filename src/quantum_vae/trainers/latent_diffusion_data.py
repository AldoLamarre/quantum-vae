"""Dataset utilities for latent diffusion training."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset


class LatentCacheDataset(Dataset):
    """Loads a precomputed {"latents": [N, D], "labels": [N]} cache produced
    by scripts/extract_latents_option_a.py (or the equivalent for another
    dataset/checkpoint). Training never touches the original images or the
    quantum circuit -- only these cached classical vectors."""

    def __init__(self, cache_path: str):
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Latent cache not found at {path}. Run the extraction script for "
                "your base checkpoint first (e.g. scripts/extract_latents_option_a.py) "
                "to precompute it before training the diffusion model."
            )
        data = torch.load(path, map_location="cpu")
        self.latents = data["latents"]
        self.labels = data["labels"]
        # Normalize to roughly unit variance -- keeps the noise schedule
        # well-matched to the data scale, standard practice for latent diffusion.
        self.mean = self.latents.mean(0, keepdim=True)
        self.std = self.latents.std(0, keepdim=True)
        self.latents_norm = (self.latents - self.mean) / self.std

    def __len__(self) -> int:
        return self.latents_norm.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"latent": self.latents_norm[idx], "label": self.labels[idx]}


def latent_diffusion_collator(batch):
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
