from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hf_vae_trainer import build_vae_dataset_bundle
from src.quantum_vae.trainers.config_parser import TrainerConfigParser
from src.quantum_vae.trainers.evaluation import evaluate_vae_reconstruction_dataset
from src.quantum_vae.utils.model_paths import registered_model_path


def _resolve_checkpoint_path(
    checkpoint: str | None,
    config: dict[str, object],
    project_root: Path,
) -> str:
    if checkpoint:
        raw = checkpoint.strip()
    else:
        raw = str(config.get("base_checkpoint") or config.get("checkpoint") or "").strip()
    if not raw:
        raise ValueError("No checkpoint provided. Set --checkpoint or base_checkpoint/checkpoint in config.")

    path = Path(raw)
    if path.is_absolute() and path.exists():
        return str(path)
    candidate = project_root / path
    if candidate.exists():
        return str(candidate)
    return registered_model_path(raw, project_root=project_root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VAE on validation/test splits with optional FID.")
    parser.add_argument("--config", type=str, required=True, help="Path to VAE JSON config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path or model registry key.")
    parser.add_argument("--split", type=str, default="both", choices=["val", "test", "both"], help="Split to evaluate.")
    parser.add_argument("--fid", action="store_true", help="Compute FID in addition to non-FID metrics.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on evaluated samples per split.")
    args = parser.parse_args()

    target_config = Path(args.config)
    if not target_config.is_absolute():
        target_config = ROOT / target_config

    cfg_parser = TrainerConfigParser(project_root=ROOT)
    config = cfg_parser.load_config(target_config)

    resolved_checkpoint = _resolve_checkpoint_path(args.checkpoint, config, ROOT)
    config["base_checkpoint"] = resolved_checkpoint

    parsed = cfg_parser.parse(config)
    model = cfg_parser.build_model(parsed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    bundle = build_vae_dataset_bundle(config)
    image_range = str(parsed.training_kwargs.get("image_range", "0_1"))
    eval_batch_size = int(parsed.training_kwargs.get("per_device_eval_batch_size", parsed.training_kwargs.get("batch_size", 32)))

    results: dict[str, dict[str, float]] = {}
    if args.split in {"val", "both"}:
        results["validation"] = evaluate_vae_reconstruction_dataset(
            model=model,
            dataset=bundle["val_set"],
            data_collator=None,
            image_range=image_range,
            batch_size=eval_batch_size,
            compute_fid=args.fid,
            sample_posterior=True,
            max_samples=args.max_samples,
        )
    if args.split in {"test", "both"} and "test_set" in bundle:
        results["test"] = evaluate_vae_reconstruction_dataset(
            model=model,
            dataset=bundle["test_set"],
            data_collator=None,
            image_range=image_range,
            batch_size=eval_batch_size,
            compute_fid=args.fid,
            sample_posterior=True,
            max_samples=args.max_samples,
        )

    output_path = Path(parsed.training_kwargs.get("output_dir", "checkpoints/vae/evaluation")) / "evaluation" / "standalone_metrics.json"
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Evaluated checkpoint: {resolved_checkpoint}")
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
