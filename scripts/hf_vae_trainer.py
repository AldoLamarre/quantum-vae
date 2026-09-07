from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path
import sys

from datasets import load_dataset
from torchvision import datasets
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import train_from_config
from src.quantum_vae.utils.cifar_family import build_cifar10_data_bundle
from src.quantum_vae.utils.imagenet_family import build_imagenet_data_bundle
from src.quantum_vae.utils.mnist_family import build_mnist_data_bundle


def build_vae_dataset_bundle(config: dict[str, object]) -> dict[str, object]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    batch_size = int(data_cfg.get("batch_size", 128))
    root = str(data_cfg.get("root", "data"))
    dataset_name = str(config.get("dataset", "mnist")).lower()

    if dataset_name == "mnist":
        training_data = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
        test_data = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())
        return build_mnist_data_bundle(training_data, test_data, batch_size=batch_size)

    if dataset_name == "cifar10":
        training_data = datasets.CIFAR10(root=root, train=True, download=True, transform=ToTensor())
        test_data = datasets.CIFAR10(root=root, train=False, download=True, transform=ToTensor())
        return build_cifar10_data_bundle(training_data, test_data, batch_size=batch_size)

    if dataset_name in {"imagenet", "imagenet-1k"}:
        ssl._create_default_https_context = ssl._create_unverified_context
        resolution = int(data_cfg.get("resolution", 256))
        dataset = load_dataset("imagenet-1k", trust_remote_code=True)
        train_dataset = dataset["train"]
        val_dataset = dataset["validation"]
        test_dataset = dataset["test"]
        transform = Compose(
            [
                Resize(resolution),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        train_dataset.set_transform(lambda example: {"pixel_values": transform(example["image"])})
        val_dataset.set_transform(lambda example: {"pixel_values": transform(example["image"])})
        test_dataset.set_transform(lambda example: {"pixel_values": transform(example["image"])})
        return build_imagenet_data_bundle(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    raise ValueError(f"Unsupported VAE dataset: {dataset_name}")


def main(config_path: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a config-driven HF VAE experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    target_config = Path(config_path) if config_path is not None else Path(args.config)
    if not target_config.is_absolute():
        target_config = ROOT / target_config

    print(f"Loading VAE config from: {target_config}")
    with target_config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    bundle = build_vae_dataset_bundle(config)
    trainer, train_results, eval_results = train_from_config(
        target_config,
        train_dataset=bundle["train_set"],
        eval_dataset=bundle["val_set"],
    )
    print(f"Trainer built: {type(trainer).__name__}")
    if train_results is not None:
        print("Training complete.")
        print(train_results)
    if eval_results is not None:
        print("Evaluation complete.")
        print(eval_results)
    if "test_set" in bundle:
        test_results = trainer.evaluate(eval_dataset=bundle["test_set"])
        print("Test complete.")
        print(test_results)


if __name__ == "__main__":
    main()
