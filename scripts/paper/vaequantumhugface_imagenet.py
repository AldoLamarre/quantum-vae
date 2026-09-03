from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import ssl
from datasets import load_dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper._paper_vae_runner import run_vae_experiment
from src.quantum_vae.utils.imagenet_family import build_imagenet_data_bundle

DEFAULT_CONFIG = ROOT / "configs" / "paper" / "vaequantumhugface_imagenet.json"


def main(config_path: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the paper ImageNet VAE experiment.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG), help="Path to the JSON config file")
    args = parser.parse_args()

    config_file = Path(config_path) if config_path is not None else Path(args.config)
    if not config_file.is_absolute():
        config_file = ROOT / config_file

    with config_file.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    ssl._create_default_https_context = ssl._create_unverified_context
    batch_size = int(config.get("data", {}).get("batch_size", 128))
    dataset = load_dataset("imagenet-1k", trust_remote_code=True)
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_dataset.set_transform(lambda example: {"pixel_values": transform(example["image"])})
    val_dataset.set_transform(lambda example: {"pixel_values": transform(example["image"])})
    test_dataset.set_transform(lambda example: {"pixel_values": transform(example["image"])})

    bundle = build_imagenet_data_bundle(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    run_vae_experiment(
        config_file,
        bundle["train_set"],
        bundle["val_set"],
        bundle["test_set"],
    )


if __name__ == "__main__":
    main()
