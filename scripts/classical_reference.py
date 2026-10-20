"""Train and evaluate a classical reference VAE (no quantum layer).

Same data, loss, schedule and final evaluation as hf_vae_trainer.py; only
the model differs (see src/quantum_vae/models/classical_vae.py).

Usage: python scripts/classical_reference.py --config configs/classical/<config>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hf_vae_trainer import build_vae_dataset_bundle  # noqa: E402
from src.quantum_vae.models.classical_vae import build_classical_vae  # noqa: E402
from src.quantum_vae.trainers.config_parser import train_from_config  # noqa: E402
from src.quantum_vae.trainers.evaluation import evaluate_vae_reconstruction_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a classical reference VAE.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    print(f"Loading classical reference config from: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    model = build_classical_vae(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ClassicalVAE: bottleneck_dim={model.bottleneck_dim}, latent_dim={model.latent_dim()}, params={n_params}")

    bundle = build_vae_dataset_bundle(config)
    trainer, train_results, eval_results = train_from_config(
        config_path,
        model=model,
        train_dataset=bundle["train_set"],
        eval_dataset=bundle["val_set"],
    )
    print(train_results)
    print(eval_results)

    if torch.cuda.is_available():
        eval_device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        eval_device = torch.device("mps")
    else:
        eval_device = torch.device("cpu")
    trainer.model.to(eval_device)

    image_range = str(getattr(trainer, "image_range", "0_1"))
    eval_batch_size = int(getattr(trainer.args, "per_device_eval_batch_size", 32))
    final_metrics = {}
    for split, key in (("validation", "val_set"), ("test", "test_set")):
        if key not in bundle:
            continue
        final_metrics[split] = evaluate_vae_reconstruction_dataset(
            model=trainer.model,
            dataset=bundle[key],
            data_collator=trainer.data_collator,
            image_range=image_range,
            batch_size=eval_batch_size,
            compute_fid=True,
            sample_posterior=True,
        )
        print(f"Final {split} metrics (with FID):")
        print(final_metrics[split])

    metrics_path = Path(trainer.args.output_dir) / "evaluation" / "full_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(f"Saved full metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
