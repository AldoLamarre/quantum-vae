"""Configuration parser and trainer factory for Quantum VAE and Classifier models."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    has_torch = True
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    has_torch = False

try:
    from transformers import TrainingArguments
    has_transformers_args = True
except ImportError:
    TrainingArguments = None  # type: ignore
    has_transformers_args = False

from .base import BaseHFQuantumTrainer, StandaloneHFTrainer, has_transformers
from .vae_trainer import QuantumVAETrainer
from .classifier_trainer import QuantumClassifierTrainer
from .data_collators import VAEDataCollator, ClassifierDataCollator
from .metrics import compute_classification_metrics, compute_vae_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TrainerParsedConfig:
    task_type: str  # "vae" | "classifier"
    model_name: str
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    training_kwargs: Dict[str, Any] = field(default_factory=dict)
    data_kwargs: Dict[str, Any] = field(default_factory=dict)
    raw_config: Dict[str, Any] = field(default_factory=dict)


class TrainerConfigParser:
    """Parses JSON / dictionary configs into models, training arguments, datasets, and trainers."""

    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        self.project_root = Path(project_root) if project_root is not None else PROJECT_ROOT

    def load_config(self, config_source: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Load configuration from a JSON file path or dictionary."""
        if isinstance(config_source, dict):
            return dict(config_source)
        path = Path(config_source)
        if not path.is_absolute():
            path = self.project_root / path
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def parse(self, config_source: Union[str, Path, Dict[str, Any]]) -> TrainerParsedConfig:
        """Parse raw configuration into a structured TrainerParsedConfig."""
        cfg = self.load_config(config_source)

        # Detect task type: VAE vs Classifier
        task_type = "classifier"
        if "family" in cfg and "vae" in str(cfg["family"]).lower():
            if "classifier" in str(cfg["family"]).lower():
                task_type = "classifier"
            elif "ablation" in str(cfg["family"]).lower():
                task_type = "classifier"
            else:
                task_type = "vae"
        elif "model" in cfg and isinstance(cfg["model"], dict) and "down_block_types" in cfg["model"]:
            task_type = "vae"
        elif "classifier" in cfg or "measurement" in cfg:
            task_type = "classifier"
        elif "kl_weight" in cfg.get("training", {}) or "tomography" in cfg:
            task_type = "vae"

        model_kwargs: Dict[str, Any] = {}
        data_kwargs: Dict[str, Any] = {}
        training_kwargs: Dict[str, Any] = {}

        if task_type == "vae":
            model_name = cfg.get("model_name", cfg.get("strategy", "amplitude"))
            if isinstance(cfg.get("model"), dict):
                model_kwargs.update(cfg["model"])
            if "tomography" in cfg:
                model_kwargs["Tomography"] = cfg["tomography"]
            if "n_qubits" in cfg:
                model_kwargs["n_qubits"] = int(cfg["n_qubits"])
            if "n_quantum_layers" in cfg:
                model_kwargs["n_quantum_layers"] = int(cfg["n_quantum_layers"])

            # Data
            if isinstance(cfg.get("data"), dict):
                data_kwargs.update(cfg["data"])
            elif "dataset" in cfg:
                data_kwargs["dataset"] = cfg["dataset"]

            # Training
            if isinstance(cfg.get("training"), dict):
                training_kwargs.update(cfg["training"])
            elif isinstance(cfg.get("trainer"), dict):
                training_kwargs.update(cfg["trainer"])

            if "output" in cfg and isinstance(cfg["output"], dict):
                training_kwargs["output_dir"] = cfg["output"].get("root", "checkpoints/vae_output")

        else:
            model_name = cfg.get("classifier", "cifar_quantum_classifier")
            # Classifier settings
            measurement_cfg = cfg.get("measurement", {})
            post_cfg = cfg.get("postprocessing_mlp", {})
            n_qubits = int(cfg.get("n_qubits", 7))
            n_layers = int(cfg.get("n_layers", 20))
            dataset_name = str(cfg.get("dataset", "cifar10")).lower()
            num_labels = 100 if dataset_name == "cifar100" else (2 if "3v5" in dataset_name else 10)
            if "num_labels" in cfg:
                num_labels = int(cfg["num_labels"])

            model_kwargs.update({
                "n_qubits": n_qubits,
                "n_layers": n_layers,
                "num_labels": num_labels,
                "measurement_kind": str(measurement_cfg.get("kind", "probability")),
                "measurement_pauli": measurement_cfg.get("pauli", "Z"),
                "postprocessing_mlp_enabled": bool(post_cfg.get("enabled", False)),
                "postprocessing_mlp_hidden_dim": int(post_cfg.get("hidden_dim", 128)),
                "softmax_enabled": bool(cfg.get("softmax", True)),
            })

            data_kwargs["dataset"] = dataset_name
            if isinstance(cfg.get("data"), dict):
                data_kwargs.update(cfg["data"])

            if isinstance(cfg.get("trainer"), dict):
                training_kwargs.update(cfg["trainer"])
            elif isinstance(cfg.get("training"), dict):
                training_kwargs.update(cfg["training"])

        # Top-level seed
        if "seed" in cfg:
            training_kwargs["seed"] = cfg["seed"]

        return TrainerParsedConfig(
            task_type=task_type,
            model_name=model_name,
            model_kwargs=model_kwargs,
            training_kwargs=training_kwargs,
            data_kwargs=data_kwargs,
            raw_config=cfg,
        )

    def build_model(self, parsed: Union[TrainerParsedConfig, Dict[str, Any]]) -> Any:
        """Instantiate the model defined in the configuration."""
        if not isinstance(parsed, TrainerParsedConfig):
            parsed = self.parse(parsed)

        if parsed.task_type == "vae":
            from src.quantum_vae.models import QuantumVAEAmplitude, QuantumVAEDataReupload

            model_name = str(parsed.model_name).lower()
            if "datareupload" in model_name or "data_reupload" in model_name or "circuit" in model_name:
                model_cls = QuantumVAEDataReupload
            else:
                model_cls = QuantumVAEAmplitude

            # Filter kwargs
            kwargs = dict(parsed.model_kwargs)
            try:
                return model_cls(**kwargs)
            except Exception:
                # Fallback to default small kwargs if needed
                fallback_kwargs = dict(
                    in_channels=kwargs.get("in_channels", 3),
                    out_channels=kwargs.get("out_channels", 3),
                    sample_size=kwargs.get("sample_size", 32),
                    block_out_channels=kwargs.get("block_out_channels", (32, 32, 64)),
                    down_block_types=kwargs.get("down_block_types", ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")),
                    up_block_types=kwargs.get("up_block_types", ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")),
                )
                return model_cls(**fallback_kwargs)

        else:
            from src.quantum_vae.models import CifarClassifierConfig, CifarQuantumClassifier

            cifar_cfg = CifarClassifierConfig(
                n_qubits=parsed.model_kwargs.get("n_qubits", 7),
                n_layers=parsed.model_kwargs.get("n_layers", 20),
                num_labels=parsed.model_kwargs.get("num_labels", 10),
                measurement_kind=parsed.model_kwargs.get("measurement_kind", "probability"),
                measurement_pauli=parsed.model_kwargs.get("measurement_pauli", "Z"),
                postprocessing_mlp_enabled=parsed.model_kwargs.get("postprocessing_mlp_enabled", False),
                postprocessing_mlp_hidden_dim=parsed.model_kwargs.get("postprocessing_mlp_hidden_dim", 128),
                classifier_enabled=parsed.model_kwargs.get("classifier_enabled", True),
                softmax_enabled=parsed.model_kwargs.get("softmax_enabled", True),
            )
            return CifarQuantumClassifier(cifar_cfg)

    def build_training_args(
        self,
        parsed: Union[TrainerParsedConfig, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Construct TrainingArguments / config object."""
        if not isinstance(parsed, TrainerParsedConfig):
            parsed = self.parse(parsed)

        t_kwargs = parsed.training_kwargs

        out_dir = output_dir or t_kwargs.get("output_dir", "checkpoints/hf_quantum_run")
        out_path = Path(out_dir)
        if not out_path.is_absolute():
            out_path = self.project_root / out_path
        out_path.mkdir(parents=True, exist_ok=True)

        epochs = int(t_kwargs.get("num_train_epochs", t_kwargs.get("epochs", 1)))
        lr = float(t_kwargs.get("learning_rate", 1e-4))
        train_bs = int(t_kwargs.get("per_device_train_batch_size", t_kwargs.get("batch_size", 32)))
        eval_bs = int(t_kwargs.get("per_device_eval_batch_size", train_bs))
        log_steps = int(t_kwargs.get("logging_steps", 25))
        seed = int(t_kwargs.get("seed", 42))

        if has_transformers and has_transformers_args:
            try:
                import inspect
                sig = inspect.signature(TrainingArguments.__init__)
                valid_params = sig.parameters.keys()

                kwargs: Dict[str, Any] = {
                    "output_dir": str(out_path),
                    "num_train_epochs": epochs,
                    "learning_rate": lr,
                    "per_device_train_batch_size": train_bs,
                    "per_device_eval_batch_size": eval_bs,
                    "logging_steps": log_steps,
                    "seed": seed,
                    "save_strategy": str(t_kwargs.get("save_strategy", "epoch")),
                    "overwrite_output_dir": bool(t_kwargs.get("overwrite_output_dir", False)),
                    "report_to": list(t_kwargs.get("report_to", ["tensorboard"])),
                    "logging_dir": str(out_path / "logs"),
                }

                eval_strat = str(t_kwargs.get("eval_strategy", t_kwargs.get("evaluation_strategy", "epoch")))
                if "eval_strategy" in valid_params:
                    kwargs["eval_strategy"] = eval_strat
                elif "evaluation_strategy" in valid_params:
                    kwargs["evaluation_strategy"] = eval_strat

                # Filter kwargs to valid_params
                final_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                return TrainingArguments(**final_kwargs)
            except Exception:
                pass

        # Standalone Training Arguments object
        class StandaloneTrainingArguments:
            def __init__(self):
                self.output_dir = str(out_path)
                self.num_train_epochs = epochs
                self.learning_rate = lr
                self.per_device_train_batch_size = train_bs
                self.per_device_eval_batch_size = eval_bs
                self.logging_steps = log_steps
                self.seed = seed
                self.save_strategy = str(t_kwargs.get("save_strategy", "epoch"))
                self.evaluation_strategy = str(t_kwargs.get("evaluation_strategy", "epoch"))
                self.overwrite_output_dir = bool(t_kwargs.get("overwrite_output_dir", False))
                self.report_to = list(t_kwargs.get("report_to", ["tensorboard"]))
                self.logging_dir = str(out_path / "logs")

        return StandaloneTrainingArguments()

    def build_trainer(
        self,
        config_source: Union[str, Path, Dict[str, Any]],
        model: Optional[Any] = None,
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        training_args: Optional[Any] = None,
        **trainer_kwargs,
    ) -> BaseHFQuantumTrainer:
        """Construct the appropriate QuantumVAETrainer or QuantumClassifierTrainer."""
        parsed = self.parse(config_source)

        if model is None:
            model = self.build_model(parsed)

        if training_args is None:
            training_args = self.build_training_args(parsed)

        if parsed.task_type == "vae":
            kl_weight = float(parsed.training_kwargs.get("kl_weight", 1e-4))
            loss_type = str(parsed.training_kwargs.get("loss_type", "mse"))
            return QuantumVAETrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                kl_weight=kl_weight,
                loss_type=loss_type,
                **trainer_kwargs,
            )
        else:
            loss_fn = parsed.training_kwargs.get("loss", "cross_entropy")
            return QuantumClassifierTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss_fn=loss_fn,
                **trainer_kwargs,
            )


def build_trainer_from_config(
    config_source: Union[str, Path, Dict[str, Any]],
    model: Optional[Any] = None,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    **kwargs,
) -> BaseHFQuantumTrainer:
    """Convenience helper to build a Hugging Face trainer from a configuration."""
    parser = TrainerConfigParser()
    return parser.build_trainer(
        config_source=config_source,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **kwargs,
    )


def train_from_config(
    config_source: Union[str, Path, Dict[str, Any]],
    model: Optional[Any] = None,
    train_dataset: Optional[Any] = None,
    eval_dataset: Optional[Any] = None,
    **kwargs,
) -> Tuple[BaseHFQuantumTrainer, Dict[str, Any], Optional[Dict[str, float]]]:
    """Load config, construct trainer, execute training, evaluate, and save model."""
    trainer = build_trainer_from_config(
        config_source=config_source,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **kwargs,
    )
    train_results = trainer.train()
    eval_results = None
    if eval_dataset is not None or getattr(trainer, "eval_dataset", None) is not None:
        eval_results = trainer.evaluate()
    trainer.save_model()
    return trainer, train_results, eval_results
