from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path
import sys
import torch

from datasets import load_dataset
from torchvision import datasets
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.trainers.config_parser import train_from_config
from src.quantum_vae.trainers.evaluation import evaluate_vae_reconstruction_dataset
from src.quantum_vae.utils.cifar_family import build_cifar10_data_bundle
from src.quantum_vae.utils.imagenet_family import build_imagenet_data_bundle
from src.quantum_vae.utils.mnist_family import build_mnist_data_bundle


IMAGENET_SPLITS = {"train": "train", "val": "validation", "test": "test"}
EVAL_SPLITS = ("val", "test")


def _imagenet_transform(data_cfg: dict[str, object]) -> Compose:
    resolution = int(data_cfg.get("resolution", 256))
    return Compose(
        [
            Resize(resolution),
            CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _load_imagenet_split(split: str, transform: Compose):
    dataset = load_dataset("imagenet-1k", split=IMAGENET_SPLITS[split], trust_remote_code=True)

    # set_transform receives batches; some ImageNet images are grayscale or CMYK.
    def to_pixel_values(batch: dict[str, list]) -> dict[str, list]:
        return {"pixel_values": [transform(image.convert("RGB")) for image in batch["image"]]}

    dataset.set_transform(to_pixel_values)
    return dataset


def build_vae_dataset_bundle(config: dict[str, object]) -> dict[str, object]:
    data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
    batch_size = int(data_cfg.get("batch_size", 128))
    root = Path(str(data_cfg.get("root", "data")))
    if not root.is_absolute():
        root = ROOT / root
    root = str(root)
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
        transform = _imagenet_transform(data_cfg)
        train_dataset, val_dataset, test_dataset = (
            _load_imagenet_split(split, transform) for split in ("train", "val", "test")
        )
        return build_imagenet_data_bundle(train_dataset, val_dataset, test_dataset, batch_size=batch_size)

    raise ValueError(f"Unsupported VAE dataset: {dataset_name}")


def build_vae_eval_datasets(config: dict[str, object], splits: tuple[str, ...]) -> dict[str, object]:
    """Return the requested evaluation datasets keyed by split ("val", "test").

    ImageNet loads only the requested splits; other datasets are small enough
    to build the full bundle. Splits a dataset does not provide are omitted.
    """
    unknown = sorted(set(splits) - set(EVAL_SPLITS))
    if unknown:
        raise ValueError(f"Unsupported evaluation splits: {unknown}")

    dataset_name = str(config.get("dataset", "mnist")).lower()
    if dataset_name in {"imagenet", "imagenet-1k"}:
        data_cfg = config.get("data", {}) if isinstance(config.get("data"), dict) else {}
        transform = _imagenet_transform(data_cfg)
        return {split: _load_imagenet_split(split, transform) for split in splits}

    bundle = build_vae_dataset_bundle(config)
    return {split: bundle[f"{split}_set"] for split in splits if f"{split}_set" in bundle}


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
    full_val_metrics = evaluate_vae_reconstruction_dataset(
        model=trainer.model,
        dataset=bundle["val_set"],
        data_collator=trainer.data_collator,
        image_range=image_range,
        batch_size=eval_batch_size,
        compute_fid=True,
        sample_posterior=True,
    )
    final_metrics["validation"] = full_val_metrics
    print("Final validation metrics (with FID):")
    print(full_val_metrics)

    if "test_set" in bundle:
        full_test_metrics = evaluate_vae_reconstruction_dataset(
            model=trainer.model,
            dataset=bundle["test_set"],
            data_collator=trainer.data_collator,
            image_range=image_range,
            batch_size=eval_batch_size,
            compute_fid=True,
            sample_posterior=True,
        )
        final_metrics["test"] = full_test_metrics
        print("Final test metrics (with FID):")
        print(full_test_metrics)

    metrics_path = Path(trainer.args.output_dir) / "evaluation" / "full_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)
    print(f"Saved full metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
