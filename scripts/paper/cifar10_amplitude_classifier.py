from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import train_from_config

DEFAULT_CONFIG = ROOT / "configs" / "hf_amplitude_classifier_family.json"


def main(config_path: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 amplitude-classifier experiment.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to the config JSON file")
    args = parser.parse_args()

    target_config = Path(config_path) if config_path is not None else Path(args.config)
    if not target_config.is_absolute():
        target_config = ROOT / target_config

    print(f"Loading CIFAR-10 amplitude-classifier config from: {target_config}")
    trainer, train_results, eval_results = train_from_config(target_config)
    print(f"Trainer built: {type(trainer).__name__}")

    if train_results is not None:
        print("Training complete.")
        print(train_results)
    if eval_results is not None:
        print("Evaluation complete.")
        print(eval_results)


if __name__ == "__main__":
    main()
