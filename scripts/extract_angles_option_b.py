"""Precompute quaternion training data for Option B (su2_angles) diffusion.

Runs the frozen data-reupload VAE's classical encoder + project_to_quantum
over real MNIST images to get the actual gate angles (quantum_input, the
30-dim = n_qubits*3 values fed to AngleEmbedding), then composes each
qubit's (x, y, z) triple into one SU(2) quaternion via quat_from_xyz.

This is the "going IN" direction described in su2_math.py: real angles ->
quaternions, used as the clean training data x0 for SU2HeatKernelSchedule.
The quantum circuit itself (qlayer) is never touched here -- only the
classical encoder and the projection layer that produces its input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import TrainerConfigParser
from src.quantum_vae.utils.model_paths import registered_model_path
from src.quantum_vae.diffusion.su2_math import quat_from_xyz


def main():
    parser_arg = argparse.ArgumentParser(description=__doc__)
    parser_arg.add_argument("--data-dir", type=str, default="data", help="MNIST root (downloaded if missing)")
    args = parser_arg.parse_args()

    cfg = json.load(open(ROOT / "configs/paper/vaequantumhugface_mnist_pretraining_data_reupload11.json"))
    cfg_parser = TrainerConfigParser()
    parsed = cfg_parser.parse(cfg)
    model = cfg_parser.build_model(parsed)

    dummy = torch.zeros(1, 1, 28, 28)
    model.initialize_projections(dummy)
    sd = torch.load(registered_model_path("mnist_datareupload_compat"), map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    n_qubits = model.n_qubits
    expected_dim = n_qubits * 3
    actual_dim = model.project_to_quantum.out_features
    if actual_dim != expected_dim:
        raise RuntimeError(
            f"project_to_quantum outputs {actual_dim} values but n_qubits*3={expected_dim} "
            f"were expected -- check this checkpoint hasn't drifted from the truncation fix "
            f"applied earlier (see the CIFAR/MNIST datareupload compat checkpoints)."
        )

    training_data = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=ToTensor())
    loader = DataLoader(training_data, batch_size=256, shuffle=False)

    all_quats = []
    all_labels = []
    with torch.no_grad():
        for x, y in loader:
            posterior = model.encode(x).latent_dist
            z = posterior.mode()
            z_flat = z.flatten(1)
            quantum_input = model.project_to_quantum(z_flat)  # (batch, n_qubits*3)
            angles = quantum_input.reshape(-1, n_qubits, 3)   # (batch, n_qubits, 3)
            quats = quat_from_xyz(angles[..., 0], angles[..., 1], angles[..., 2])  # (batch, n_qubits, 4)
            all_quats.append(quats)
            all_labels.append(y)

    quaternions = torch.cat(all_quats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    print("quaternions shape:", quaternions.shape)
    print("unit-norm check (should be ~1.0):", quaternions.norm(dim=-1).mean().item())

    out_dir = ROOT / "diffusion_data"
    out_dir.mkdir(exist_ok=True)
    torch.save({"quaternions": quaternions, "labels": labels}, out_dir / "mnist_quaternions_option_b.pt")
    print(f"saved {out_dir / 'mnist_quaternions_option_b.pt'}")


if __name__ == "__main__":
    main()

