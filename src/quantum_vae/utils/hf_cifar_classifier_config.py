from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.quantum_vae.models import CifarClassifierConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "hf_cifar_classifier_family.json"


@dataclass(frozen=True)
class HFCifarModelConfig:
    dataset: str
    classifier: str
    measurement_kind: str
    measurement_pauli: Optional[str]
    softmax_enabled: bool
    postprocessing_mlp_enabled: bool
    postprocessing_mlp_hidden_dim: int


@dataclass(frozen=True)
class HFTrainingConfig:
    output_dir: str
    logging_dir: str
    overwrite_output_dir: bool
    logging_strategy: str
    logging_steps: int
    save_strategy: str
    save_total_limit: Optional[int]
    evaluation_strategy: str
    load_best_model_at_end: bool
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    num_train_epochs: int
    report_to: list[str]


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dataset_name(config: Dict[str, Any]) -> str:
    dataset_name = str(config.get("dataset", "cifar10")).lower()
    return "cifar100" if dataset_name == "cifar100" else "cifar10"


def build_model_config(config: Dict[str, Any]) -> HFCifarModelConfig:
    measurement_cfg = config.get("measurement", {})
    measurement_kind = str(measurement_cfg.get("kind", "probability")).lower()
    measurement_pauli: Optional[str] = None
    if measurement_kind == "expectation":
        measurement_pauli = str(measurement_cfg.get("pauli", "Z")).upper()

    post_cfg = config.get("postprocessing_mlp", {})
    return HFCifarModelConfig(
        dataset=_dataset_name(config),
        classifier=str(config.get("classifier", "pretrained")).lower(),
        measurement_kind=measurement_kind,
        measurement_pauli=measurement_pauli,
        softmax_enabled=bool(config.get("softmax", True)),
        postprocessing_mlp_enabled=bool(post_cfg.get("enabled", False)),
        postprocessing_mlp_hidden_dim=int(post_cfg.get("hidden_dim", 128)),
    )


def build_classifier_config(config: Dict[str, Any]) -> CifarClassifierConfig:
    model_cfg = build_model_config(config)
    n_qubits = int(config.get("n_qubits", 7))
    n_layers = int(config.get("n_layers", 20))
    num_labels = 100 if model_cfg.dataset == "cifar100" else 10
    return CifarClassifierConfig(
        n_qubits=n_qubits,
        n_layers=n_layers,
        num_labels=num_labels,
        measurement_kind=model_cfg.measurement_kind,
        measurement_pauli=model_cfg.measurement_pauli,
        postprocessing_mlp_enabled=model_cfg.postprocessing_mlp_enabled,
        postprocessing_mlp_hidden_dim=model_cfg.postprocessing_mlp_hidden_dim,
        softmax_enabled=model_cfg.softmax_enabled,
    )


def resolve_output_dir(config: Dict[str, Any], project_root: Optional[str | Path] = None) -> Path:
    project_root_path = Path(project_root) if project_root is not None else PROJECT_ROOT
    trainer_cfg = config.get("trainer", {})
    output_dir = project_root_path / trainer_cfg.get("output_dir", "checkpoints/hf_cifar_classifier")

    if bool(config.get("checkpoint", False)) and output_dir.exists() and not bool(
        trainer_cfg.get("overwrite_output_dir", False)
    ):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output_dir = output_dir.parent / f"{output_dir.name}-{timestamp}"

    return output_dir


def build_training_args(config: Dict[str, Any], project_root: Optional[str | Path] = None) -> HFTrainingConfig:
    trainer_cfg = config.get("trainer", {})
    output_dir = resolve_output_dir(config, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging_dir = Path(trainer_cfg.get("logging_dir", str(output_dir / "logs")))
    if not logging_dir.is_absolute():
        root = Path(project_root) if project_root is not None else PROJECT_ROOT
        logging_dir = root / logging_dir
    logging_dir.mkdir(parents=True, exist_ok=True)

    save_strategy = str(trainer_cfg.get("save_strategy", "epoch"))
    save_total_limit = trainer_cfg.get("save_total_limit")
    if not bool(config.get("checkpoint", False)):
        save_strategy = "no"
        save_total_limit = None

    return HFTrainingConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        overwrite_output_dir=bool(trainer_cfg.get("overwrite_output_dir", False)),
        logging_strategy=str(trainer_cfg.get("logging_strategy", "steps")),
        logging_steps=int(trainer_cfg.get("logging_steps", 25)),
        save_strategy=save_strategy,
        save_total_limit=int(save_total_limit) if save_total_limit is not None else None,
        evaluation_strategy=str(trainer_cfg.get("evaluation_strategy", "epoch")),
        load_best_model_at_end=bool(trainer_cfg.get("load_best_model_at_end", bool(config.get("checkpoint", False)))),
        per_device_train_batch_size=int(trainer_cfg.get("per_device_train_batch_size", 32)),
        per_device_eval_batch_size=int(trainer_cfg.get("per_device_eval_batch_size", 32)),
        learning_rate=float(trainer_cfg.get("learning_rate", 1e-4)),
        num_train_epochs=int(trainer_cfg.get("num_train_epochs", 1)),
        report_to=list(trainer_cfg.get("report_to", ["tensorboard"])),
    )


def main(config_path: Optional[str | Path] = None) -> None:
    import argparse
    from src.quantum_vae.trainers import TrainerConfigParser, build_trainer_from_config

    parser = argparse.ArgumentParser(description="Hugging Face Quantum Model Trainer")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to config JSON")
    args, unknown = parser.parse_known_args()

    target_config = config_path if config_path is not None else args.config
    print(f"Loading config from: {target_config}")

    config_parser = TrainerConfigParser()
    parsed = config_parser.parse(target_config)
    print(f"Task type: {parsed.task_type}, Model: {parsed.model_name}")

    trainer = build_trainer_from_config(target_config)
    print(f"Trainer constructed: {type(trainer).__name__}")
