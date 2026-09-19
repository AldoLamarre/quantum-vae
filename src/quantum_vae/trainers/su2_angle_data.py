"""Dataset for Option B (su2_angles): loads a precomputed
{"quaternions": [N, n_qubits, 4], "labels": [N]} cache.

Deliberately separate from latent_diffusion_data.LatentCacheDataset, which
per-channel-normalizes its data -- correct for Option A's unconstrained
z_flat vectors, but wrong here: quaternions are already unit-norm points on
SU(2), and renormalizing them the way LatentCacheDataset does would push
them off the group entirely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset


class QuaternionCacheDataset(Dataset):
    """Loads a precomputed quaternion cache produced by
    scripts/extract_angles_option_b.py. No normalization is applied --
    the data is already unit-norm quaternions, and SU2HeatKernelSchedule's
    tangent-space noising operates directly on that representation."""

    def __init__(self, cache_path: str):
        path = Path(cache_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Quaternion cache not found at {path}. Run "
                "scripts/extract_angles_option_b.py first to precompute it."
            )
        data = torch.load(path, map_location="cpu")
        self.quaternions = data["quaternions"]  # (N, n_qubits, 4)
        self.labels = data["labels"]

    def __len__(self) -> int:
        return self.quaternions.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"latent": self.quaternions[idx], "label": self.labels[idx]}


def quaternion_diffusion_collator(batch):
    return {
        "latent": torch.stack([b["latent"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
