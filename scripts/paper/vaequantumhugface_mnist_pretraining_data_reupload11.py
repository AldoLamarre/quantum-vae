from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torchvision import datasets
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper._paper_vae_runner import run_vae_experiment
from src.quantum_vae.utils.mnist_family import build_mnist_data_bundle

DEFAULT_CONFIG = ROOT / "configs" / "paper" / "vaequantumhugface_mnist_pretraining_data_reupload11.json"


def main(config_path: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the paper MNIST data-reupload VAE experiment.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to the JSON config file")
    args = parser.parse_args()

    config_file = Path(config_path) if config_path is not None else Path(args.config)
    if not config_file.is_absolute():
        config_file = ROOT / config_file

    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    batch_size = int(config.get("data", {}).get("batch_size", 32))
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    bundle = build_mnist_data_bundle(training_data, test_data, batch_size=batch_size)

    run_vae_experiment(
        config_file,
        bundle["train_set"],
        bundle["val_set"],
        bundle["test_set"],
    )


if __name__ == "__main__":
    main()
