from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.quantum_vae.models.amplitude_classifier import ClassifierPipelineConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "hf_amplitude_classifier_family.json"


@dataclass(frozen=True)
class VAEBackboneConfig:
    strategy: str  # ansatz_vae | amplitude_vae
    checkpoint: Optional[str]
    vae_class: str
    freeze_classical_parts: bool
    train_quantum_parts: bool
    train_projection_layers: bool
    ansatz_name: Optional[str]
    n_qubits: Optional[int]
    n_quantum_layers: Optional[int]


@dataclass(frozen=True)
class HFAmplitudeClassifierModelConfig:
    dataset: str
    classifier_mode: str  # ansatz | amplitude
    vae_backbone: VAEBackboneConfig
    measurement_kind: str
    measurement_pauli: Optional[str]
    softmax_enabled: bool
    num_labels: int
    n_qubits: int
    n_layers: int
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
    return str(config.get("dataset", "cifar10")).lower()


def _resolve_softmax_enabled(config: Dict[str, Any]) -> bool:
    classifier_cfg = config.get("classifier")
    if isinstance(classifier_cfg, dict) and "softmax" in classifier_cfg:
        return bool(classifier_cfg.get("softmax", True))
    return bool(config.get("softmax", True))


def _parse_vae_backbone(config: Dict[str, Any]) -> VAEBackboneConfig:
    vae_cfg = config.get("vae_backbone")
    if not isinstance(vae_cfg, dict):
        vae_cfg = config.get("base_vae", {})

    strategy = str(vae_cfg.get("strategy", "ansatz_vae")).lower()
    if strategy not in {"ansatz_vae", "amplitude_vae"}:
        raise ValueError("vae_backbone.strategy must be 'ansatz_vae' or 'amplitude_vae'.")

    strategy_cfg = vae_cfg.get(strategy, {})
    if not isinstance(strategy_cfg, dict):
        strategy_cfg = {}

    checkpoint = strategy_cfg.get("checkpoint", vae_cfg.get("checkpoint"))
    default_vae_class = "QuantumVAEDataReupload" if strategy == "ansatz_vae" else "QuantumVAEAmplitude"
    vae_class = str(strategy_cfg.get("vae_class", vae_cfg.get("vae_class", default_vae_class)))
    freeze_classical_parts = bool(strategy_cfg.get("freeze_classical_parts", True))
    train_quantum_parts = bool(strategy_cfg.get("train_quantum_parts", True))
    train_projection_layers = bool(strategy_cfg.get("train_projection_layers", True))
    ansatz_name = strategy_cfg.get("name")
    n_qubits = strategy_cfg.get("n_qubits", vae_cfg.get("n_qubits"))
    n_quantum_layers = strategy_cfg.get("n_quantum_layers", vae_cfg.get("n_quantum_layers"))
    return VAEBackboneConfig(
        strategy=strategy,
        checkpoint=str(checkpoint) if checkpoint is not None else None,
        vae_class=vae_class,
        freeze_classical_parts=freeze_classical_parts,
        train_quantum_parts=train_quantum_parts,
        train_projection_layers=train_projection_layers,
        ansatz_name=str(ansatz_name) if ansatz_name is not None else None,
        n_qubits=int(n_qubits) if n_qubits is not None else None,
        n_quantum_layers=int(n_quantum_layers) if n_quantum_layers is not None else None,
    )


def build_model_config(config: Dict[str, Any]) -> HFAmplitudeClassifierModelConfig:
    measurement_cfg = config.get("measurement", {})
    measurement_kind = str(measurement_cfg.get("kind", "probability")).lower()
    measurement_pauli: Optional[str] = None
    if measurement_kind == "expectation":
        measurement_pauli = str(measurement_cfg.get("pauli", "Z")).upper()

    post_cfg = config.get("postprocessing_mlp", {})
    classifier_mode = str(config.get("classifier_mode", "ansatz")).lower()
    if classifier_mode not in {"ansatz", "amplitude"}:
        raise ValueError("classifier_mode must be 'ansatz' or 'amplitude'.")

    softmax_enabled = _resolve_softmax_enabled(config)
    if not softmax_enabled:
        raise ValueError("classifier.softmax=false is not implemented yet; set classifier.softmax=true.")
    num_labels_cfg = config.get("classifier", {})
    if isinstance(num_labels_cfg, dict) and "num_labels" in num_labels_cfg:
        num_labels = int(num_labels_cfg["num_labels"])
    else:
        num_labels = int(config.get("num_labels", 100 if _dataset_name(config) == "cifar100" else 10))

    vae_backbone = _parse_vae_backbone(config)
    n_qubits = int(config.get("n_qubits", vae_backbone.n_qubits or 7))
    n_layers = int(config.get("n_layers", vae_backbone.n_quantum_layers or 20))

    return HFAmplitudeClassifierModelConfig(
        dataset=_dataset_name(config),
        classifier_mode=classifier_mode,
        vae_backbone=vae_backbone,
        measurement_kind=measurement_kind,
        measurement_pauli=measurement_pauli,
        softmax_enabled=softmax_enabled,
        num_labels=num_labels,
        n_qubits=n_qubits,
        n_layers=n_layers,
        postprocessing_mlp_enabled=bool(post_cfg.get("enabled", False)),
        postprocessing_mlp_hidden_dim=int(post_cfg.get("hidden_dim", 128)),
    )


def build_classifier_config(config: Dict[str, Any]) -> ClassifierPipelineConfig:
    model_cfg = build_model_config(config)
    return ClassifierPipelineConfig(
        classifier_mode=model_cfg.classifier_mode,
        n_qubits=model_cfg.n_qubits,
        n_layers=model_cfg.n_layers,
        num_labels=model_cfg.num_labels,
        measurement_kind=model_cfg.measurement_kind,
        measurement_pauli=model_cfg.measurement_pauli,
        postprocessing_mlp_enabled=model_cfg.postprocessing_mlp_enabled,
        postprocessing_mlp_hidden_dim=model_cfg.postprocessing_mlp_hidden_dim,
        softmax_enabled=model_cfg.softmax_enabled,
    )


def resolve_output_dir(config: Dict[str, Any], project_root: Optional[str | Path] = None) -> Path:
    project_root_path = Path(project_root) if project_root is not None else PROJECT_ROOT
    trainer_cfg = config.get("trainer", {})
    output_dir = project_root_path / trainer_cfg.get("output_dir", "checkpoints/hf_amplitude_classifier")

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


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "VAEBackboneConfig",
    "HFAmplitudeClassifierModelConfig",
    "HFTrainingConfig",
    "load_config",
    "build_model_config",
    "build_classifier_config",
    "resolve_output_dir",
    "build_training_args",
    "main",
]
