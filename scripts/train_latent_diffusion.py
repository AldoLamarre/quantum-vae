"""Train a latent diffusion model from a JSON config, e.g.:

    python scripts/train_latent_diffusion.py \
        --config configs/paper/vaequantumhugface_mnist_latent_diffusion.json

Mirrors scripts/hf_vae_trainer.py's shape: load config, build trainer via
TrainerConfigParser, train, save. Requires the latent cache referenced by
the config's data.latents_cache to already exist -- run
scripts/extract_latents_option_a.py first if it doesn't.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import TrainerConfigParser


def main(config_path: str | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a config-driven latent diffusion experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    target_config = Path(config_path) if config_path is not None else Path(args.config)
    if not target_config.is_absolute():
        target_config = ROOT / target_config

    print(f"Loading latent diffusion config from: {target_config}")
    with target_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    cfg_parser = TrainerConfigParser()
    trainer = cfg_parser.build_trainer(config)
    print(f"Trainer built: {type(trainer).__name__}, denoiser: {type(trainer.model).__name__}")

    train_output = trainer.train()
    print("Training complete.")
    print(getattr(train_output, "metrics", train_output))

    trainer.save_model()
    print(f"Saved denoiser to {trainer.args.output_dir}")


if __name__ == "__main__":
    main()
