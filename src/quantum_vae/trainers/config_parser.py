"""Configuration parser and trainer factory for Quantum VAE and Classifier models."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import TrainingArguments

from .base import BaseHFQuantumTrainer
from .vae_trainer import QuantumVAETrainer
from .classifier_trainer import QuantumClassifierTrainer
from src.quantum_vae.utils.hf_classifier_config import (
    build_model_config as build_classifier_model_config,
    build_vae_backbone_instance,
)
from src.quantum_vae.utils.model_paths import registered_model_path

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

        # Detect task type: VAE vs Classifier.
        # An explicit "task_type" field always wins. Otherwise, fall back to
        # heuristics based on config shape/naming -- but if none of those
        # match either, raise rather than silently assuming "classifier".
        explicit_task_type = cfg.get("task_type")
        if explicit_task_type is not None:
            task_type = str(explicit_task_type).lower()
            if task_type not in ("vae", "classifier", "latent_diffusion"):
                raise ValueError(
                    f"Unsupported task_type '{explicit_task_type}' in config. "
                    "Expected 'vae', 'classifier', or 'latent_diffusion'."
                )
        else:
            task_type = None
            if "family" in cfg and "vae" in str(cfg["family"]).lower():
                if "classifier" in str(cfg["family"]).lower():
                    task_type = "classifier"
                elif "ablation" in str(cfg["family"]).lower():
                    task_type = "classifier"
                else:
                    task_type = "vae"
            elif "model" in cfg and isinstance(cfg["model"], dict) and "down_block_types" in cfg["model"]:
                task_type = "vae"
            elif "classifier" in cfg or "measurement" in cfg or "classifier_mode" in cfg:
                task_type = "classifier"
            elif "kl_weight" in cfg.get("training", {}):
                task_type = "vae"

            if task_type is None:
                raise ValueError(
                    "Could not determine task_type ('vae' or 'classifier') from this "
                    "config -- none of the usual signals (family name, model."
                    "down_block_types, classifier/measurement/classifier_mode keys, "
                    "training.kl_weight) matched. Add an explicit \"task_type\": \"vae\" "
                    "or \"task_type\": \"classifier\" field to the config to resolve this."
                )

        model_kwargs: Dict[str, Any] = {}
        data_kwargs: Dict[str, Any] = {}
        training_kwargs: Dict[str, Any] = {}

        if task_type == "vae":
            model_name = cfg.get("model_name", cfg.get("strategy", "amplitude"))
            if isinstance(cfg.get("model"), dict):
                model_kwargs.update(cfg["model"])
            if "tomography" in cfg:
                raise ValueError(
                    "'tomography' is no longer a supported config option (it was a "
                    "deprecated MNIST-only ablation that doesn't generalize across "
                    "dimensions). Remove it from the config. If you need it for "
                    "debugging a specific model instance, call model.set_Tomo(...) "
                    "directly instead."
                )
            if "n_qubits" in cfg:
                model_kwargs["n_qubits"] = int(cfg["n_qubits"])
            if "n_quantum_layers" in cfg:
                model_kwargs["n_quantum_layers"] = int(cfg["n_quantum_layers"])

            # Neutral-atom pulse variant: device-level keys (see
            # NeutralAtomDeviceConfig / build_model's is_pulse branch).
            if "n_atoms" in cfg:
                model_kwargs["n_atoms"] = int(cfg["n_atoms"])
            if "register_geometry" in cfg:
                model_kwargs["register_geometry"] = str(cfg["register_geometry"])
            if "atom_spacing_um" in cfg:
                model_kwargs["atom_spacing_um"] = float(cfg["atom_spacing_um"])
            if "r0_um" in cfg:
                model_kwargs["r0_um"] = float(cfg["r0_um"])
            if "C6" in cfg:
                model_kwargs["C6"] = float(cfg["C6"])
            if "evolution_time_us" in cfg:
                model_kwargs["evolution_time_us"] = float(cfg["evolution_time_us"])
            if "n_segments" in cfg:
                model_kwargs["n_segments"] = int(cfg["n_segments"])
            if "measurement_kind" in cfg:
                model_kwargs["measurement_kind"] = str(cfg["measurement_kind"])
            if "correlator_order" in cfg:
                model_kwargs["correlator_order"] = int(cfg["correlator_order"])
            if "n_clusters" in cfg:
                model_kwargs["n_clusters"] = int(cfg["n_clusters"])
            if "cluster_routing" in cfg:
                model_kwargs["cluster_routing"] = str(cfg["cluster_routing"])

            # Data
            if isinstance(cfg.get("data"), dict):
                data_kwargs.update(cfg["data"])
            elif "dataset" in cfg:
                data_kwargs["dataset"] = cfg["dataset"]

            # data.batch_size is the one field to set for batch size -- seeded
            # here as training_kwargs' base value, before the training/trainer
            # merges below, so it applies to both train and eval (per_device_eval
            # defaults to per_device_train when unset) unless a config
            # explicitly overrides one of them via trainer.per_device_*_batch_size.
            if "batch_size" in data_kwargs:
                training_kwargs["batch_size"] = data_kwargs["batch_size"]

            # Training
            if isinstance(cfg.get("training"), dict):
                training_kwargs.update(cfg["training"])
            if isinstance(cfg.get("trainer"), dict):
                training_kwargs.update(cfg["trainer"])

            image_range = training_kwargs.get("image_range", cfg.get("image_range"))
            if image_range is None:
                dataset_name = str(cfg.get("dataset", data_kwargs.get("dataset", "mnist"))).lower()
                image_range = "-1_1" if "imagenet" in dataset_name else "0_1"
            training_kwargs["image_range"] = str(image_range)

            if "output" in cfg and isinstance(cfg["output"], dict):
                default_root = f"checkpoints/vae/{cfg.get('family', cfg.get('model_name', 'run'))}"
                training_kwargs["output_dir"] = cfg["output"].get("root", default_root)

        elif task_type == "latent_diffusion":
            model_name = cfg.get("model_name", "latent_diffusion")
            if isinstance(cfg.get("model"), dict):
                model_kwargs.update(cfg["model"])

            # The frozen base VAE this diffuser trains on top of -- reuses the
            # exact same base_checkpoint/registry resolution as the vae branch,
            # just under a distinctly-named key so both can coexist in one config.
            base_checkpoint = cfg.get("base_checkpoint")
            if not base_checkpoint:
                raise ValueError(
                    "task_type='latent_diffusion' configs must specify 'base_checkpoint' "
                    "(a registry key or path to the frozen VAE this diffuser decodes through) "
                    "and 'base_model_config' (the config used to build that VAE)."
                )
            base_model_config = cfg.get("base_model_config")
            if not base_model_config:
                raise ValueError(
                    "task_type='latent_diffusion' configs must specify 'base_model_config' "
                    "(path to the JSON config that builds the frozen base VAE)."
                )
            model_kwargs["base_checkpoint"] = str(base_checkpoint)
            model_kwargs["base_model_config"] = str(base_model_config)

            if isinstance(cfg.get("data"), dict):
                data_kwargs.update(cfg["data"])
            if "latents_cache" not in data_kwargs:
                raise ValueError(
                    "task_type='latent_diffusion' configs must specify 'data.latents_cache' "
                    "(path to a precomputed {latents, labels} .pt file -- see "
                    "scripts/extract_latents_option_a.py)."
                )
            if "batch_size" in data_kwargs:
                training_kwargs["batch_size"] = data_kwargs["batch_size"]

            if isinstance(cfg.get("training"), dict):
                training_kwargs.update(cfg["training"])
            if isinstance(cfg.get("trainer"), dict):
                training_kwargs.update(cfg["trainer"])

        else:
            classifier_cfg = build_classifier_model_config(cfg)
            model_name = str(classifier_cfg.classifier_mode)
            model_kwargs.update({
                "classifier_mode": classifier_cfg.classifier_mode,
                "n_qubits": classifier_cfg.n_qubits,
                "n_layers": classifier_cfg.n_layers,
                "num_labels": classifier_cfg.num_labels,
                "measurement_kind": classifier_cfg.measurement_kind,
                "measurement_pauli": classifier_cfg.measurement_pauli,
                "postprocessing_mlp_enabled": classifier_cfg.postprocessing_mlp_enabled,
                "postprocessing_mlp_hidden_dim": classifier_cfg.postprocessing_mlp_hidden_dim,
                "logits": classifier_cfg.logits,
                "softmax_enabled": classifier_cfg.softmax_enabled,
                "vae_backbone": cfg.get("vae_backbone", {}),
            })

            data_kwargs["dataset"] = classifier_cfg.dataset
            if isinstance(cfg.get("data"), dict):
                data_kwargs.update(cfg["data"])

            # Same single-source-of-truth pattern as the vae branch above.
            if "batch_size" in data_kwargs:
                training_kwargs["batch_size"] = data_kwargs["batch_size"]

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
            from src.quantum_vae.models.quantum_vae_amplitude import QuantumVAEAmplitude
            from src.quantum_vae.models.quantum_vae_datareupload import QuantumVAEDataReupload
            from src.quantum_vae.models.quantum_vae_neutral_atom import NeutralAtomDeviceConfig, QuantumVAENeutralAtom

            model_name = str(parsed.model_name).lower()
            pulse_keywords = ("neutral_atom", "neutral atom", "rydberg", "pulse", "analog")
            is_pulse = any(kw in model_name for kw in pulse_keywords)
            if is_pulse:
                model_cls = QuantumVAENeutralAtom
            elif "datareupload" in model_name or "data_reupload" in model_name or "circuit" in model_name:
                model_cls = QuantumVAEDataReupload
            else:
                model_cls = QuantumVAEAmplitude

            # Filter kwargs
            kwargs = dict(parsed.model_kwargs)

            if is_pulse:
                # QuantumVAENeutralAtom takes a NeutralAtomDeviceConfig, not
                # flat kwargs -- pull device-level keys out of kwargs first,
                # everything else (in_channels, sample_size, ...) still
                # flows through to AutoencoderKL as usual.
                device_keys = (
                    "n_atoms", "register_geometry", "atom_spacing_um",
                    "r0_um", "C6", "evolution_time_us", "n_segments",
                    "measurement_kind", "correlator_order",
                    "n_clusters", "cluster_routing",
                )
                device_kwargs = {k: kwargs.pop(k) for k in device_keys if k in kwargs}
                if "n_atoms" not in device_kwargs:
                    raise ValueError(
                        "Pulse VAE config (model_name matching "
                        f"{pulse_keywords}) must specify 'n_atoms'."
                    )
                device_cfg = NeutralAtomDeviceConfig.from_geometry(**device_kwargs)
                try:
                    model = model_cls(device_cfg, **kwargs)
                except Exception as exc:
                    fallback_kwargs = dict(
                        in_channels=kwargs.get("in_channels", 3),
                        out_channels=kwargs.get("out_channels", 3),
                        sample_size=kwargs.get("sample_size", 32),
                        block_out_channels=kwargs.get("block_out_channels", (32, 32, 64)),
                        down_block_types=kwargs.get("down_block_types", ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")),
                        up_block_types=kwargs.get("up_block_types", ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")),
                    )
                    warnings.warn(
                        f"Failed to construct {model_cls.__name__} with configured kwargs "
                        f"{kwargs}: {type(exc).__name__}: {exc}. Falling back to default "
                        f"architecture kwargs {fallback_kwargs}. The model you get may NOT "
                        "match what your config requested -- fix the underlying error above "
                        "if that matters for this run.",
                        stacklevel=2,
                    )
                    model = model_cls(device_cfg, **fallback_kwargs)
            else:
                try:
                    model = model_cls(**kwargs)
                except Exception as exc:
                    # Fallback to default small kwargs if needed
                    fallback_kwargs = dict(
                        in_channels=kwargs.get("in_channels", 3),
                        out_channels=kwargs.get("out_channels", 3),
                        sample_size=kwargs.get("sample_size", 32),
                        block_out_channels=kwargs.get("block_out_channels", (32, 32, 64)),
                        down_block_types=kwargs.get("down_block_types", ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D")),
                        up_block_types=kwargs.get("up_block_types", ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D")),
                    )
                    warnings.warn(
                        f"Failed to construct {model_cls.__name__} with configured kwargs "
                        f"{kwargs}: {type(exc).__name__}: {exc}. Falling back to default "
                        f"architecture kwargs {fallback_kwargs}. The model you get may NOT "
                        "match what your config requested -- fix the underlying error above "
                        "if that matters for this run.",
                        stacklevel=2,
                    )
                    model = model_cls(**fallback_kwargs)

                
            # IMPORTANT: force any lazily-created layers (e.g. project_to_quantum /
            # project_from_quantum) to exist NOW, before checkpoint loading and before
            # the trainer builds an optimizer from model.parameters(). Without this,
            # those layers are created on the first forward() call inside the training
            # loop -- which happens *after* the optimizer has already snapshotted the
            # parameter list -- so they silently never get trained (stay at random init).
            if hasattr(model, "initialize_projections") and getattr(model, "project_to_quantum", "sentinel") is None:
                dummy_channels = kwargs.get("in_channels", 3)
                dummy_size = kwargs.get("sample_size", 32)
                dummy_input = torch.zeros(1, dummy_channels, dummy_size, dummy_size)
                with torch.no_grad():
                    model.initialize_projections(dummy_input)

            checkpoint = parsed.raw_config.get("base_checkpoint", parsed.raw_config.get("checkpoint"))
            if isinstance(checkpoint, str) and checkpoint.strip():
                checkpoint_value = checkpoint.strip()
                checkpoint_path = Path(checkpoint_value)
                registry_error: Optional[Exception] = None
                if not checkpoint_path.is_absolute():
                    candidate = self.project_root / checkpoint_path
                    if candidate.exists():
                        checkpoint_path = candidate
                    else:
                        try:
                            checkpoint_path = Path(registered_model_path(checkpoint_value, project_root=self.project_root))
                        except Exception as exc:
                            # checkpoint_value isn't a registry key either; fall
                            # through to the explicit existence check below,
                            # which will raise with a clear message.
                            registry_error = exc
                            checkpoint_path = candidate
                if not checkpoint_path.exists():
                    hint = f" (registry lookup also failed: {registry_error})" if registry_error is not None else ""
                    raise FileNotFoundError(
                        f"Checkpoint '{checkpoint_value}' was requested in the config but could not be "
                        f"resolved to an existing file (looked for: {checkpoint_path}){hint}. "
                        "Fix the path/registry key, or remove base_checkpoint/checkpoint from the "
                        "config to train from scratch."
                    )
                state = torch.load(checkpoint_path, map_location="cpu")
                if isinstance(state, dict):
                    if isinstance(state.get("state_dict"), dict):
                        state = state["state_dict"]
                    elif isinstance(state.get("model_state_dict"), dict):
                        state = state["model_state_dict"]
                load_result = model.load_state_dict(state, strict=False)
                missing = getattr(load_result, "missing_keys", [])
                unexpected = getattr(load_result, "unexpected_keys", [])
                if missing or unexpected:
                    warnings.warn(
                        f"Loaded checkpoint '{checkpoint_path}' with mismatched keys: "
                        f"{len(missing)} missing, {len(unexpected)} unexpected. "
                        f"missing_keys={missing[:10]}{'...' if len(missing) > 10 else ''}, "
                        f"unexpected_keys={unexpected[:10]}{'...' if len(unexpected) > 10 else ''}. "
                        "The model may be partially randomly initialized.",
                        stacklevel=2,
                    )
            return model

        elif parsed.task_type == "latent_diffusion":
            model_kwargs = dict(parsed.model_kwargs)
            variant = str(parsed.model_name).lower()

            if variant == "su2_angles":
                from src.quantum_vae.diffusion.su2_angles import AngleSlotDenoiser

                return AngleSlotDenoiser(
                    n_qubits=int(model_kwargs.get("n_qubits", 10)),
                    n_classes=int(model_kwargs.get("n_classes", 10)),
                    hidden=int(model_kwargs.get("hidden", 512)),
                    n_timesteps=int(model_kwargs.get("n_timesteps", 1000)),
                )

            from src.quantum_vae.diffusion.euclidean import FlatLatentDenoiser

            denoiser = FlatLatentDenoiser(
                latent_dim=int(model_kwargs.get("latent_dim", 196)),
                n_classes=int(model_kwargs.get("n_classes", 10)),
                hidden=int(model_kwargs.get("hidden", 512)),
                n_timesteps=int(model_kwargs.get("n_timesteps", 1000)),
            )
            # Deliberately NOT attaching the base VAE here: nn.Module's
            # __setattr__ auto-registers any Module-valued attribute as a
            # trainable submodule, which would silently defeat "frozen" (it'd
            # get saved into every checkpoint, moved by .to(), etc.).
            # build_trainer() resolves the frozen base VAE independently.
            return denoiser

        else:
            from src.quantum_vae.models.amplitude_classifier import AmplitudeClassifierPipeline
            from src.quantum_vae.models.classifier_base import ClassifierPipelineConfig
            from src.quantum_vae.models.datareupload_classifier import DataReuploadClassifierPipeline
            from src.quantum_vae.models.neutral_atom_classifier import NeutralAtomClassifierPipeline

            classifier_cfg = ClassifierPipelineConfig(
                classifier_mode=parsed.model_kwargs.get("classifier_mode", "ansatz"),
                n_qubits=parsed.model_kwargs.get("n_qubits", 7),
                n_layers=parsed.model_kwargs.get("n_layers", 20),
                num_labels=parsed.model_kwargs.get("num_labels", 10),
                measurement_kind=parsed.model_kwargs.get("measurement_kind", "probability"),
                measurement_pauli=parsed.model_kwargs.get("measurement_pauli", "Z"),
                postprocessing_mlp_enabled=parsed.model_kwargs.get("postprocessing_mlp_enabled", False),
                postprocessing_mlp_hidden_dim=parsed.model_kwargs.get("postprocessing_mlp_hidden_dim", 128),
                logits=parsed.model_kwargs.get("logits", parsed.model_kwargs.get("softmax_enabled", True)),
                softmax_enabled=parsed.model_kwargs.get("softmax_enabled", None),
            )
            mode = str(classifier_cfg.classifier_mode).lower()
            backbone_cfg = parsed.raw_config.get("vae_backbone")
            backbone_instance = None
            if isinstance(backbone_cfg, dict) and backbone_cfg:
                backbone_instance = build_vae_backbone_instance(parsed.raw_config, project_root=self.project_root)
            if mode == "amplitude":
                model = AmplitudeClassifierPipeline(classifier_cfg, vae_backbone_instance=backbone_instance)
            elif mode == "neutral_atom":
                model = NeutralAtomClassifierPipeline(classifier_cfg, vae_backbone_instance=backbone_instance)
            else:
                model = DataReuploadClassifierPipeline(classifier_cfg, vae_backbone_instance=backbone_instance)

            if backbone_instance is not None:
                model.set_vae_backbone(backbone_instance)
            return model

    def build_training_args(
        self,
        parsed: Union[TrainerParsedConfig, Dict[str, Any]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Construct TrainingArguments / config object."""
        if not isinstance(parsed, TrainerParsedConfig):
            parsed = self.parse(parsed)

        t_kwargs = parsed.training_kwargs

        family = parsed.raw_config.get("family", parsed.raw_config.get("model_name", parsed.raw_config.get("name", "run")))
        task_dir = {"vae": "vae", "latent_diffusion": "diffusion"}.get(parsed.task_type, "classifier")
        default_out_dir = f"checkpoints/{task_dir}/{family}"
        out_dir = output_dir or t_kwargs.get("output_dir", default_out_dir)
        # One timestamp per run, applied to the whole output_dir -- so
        # checkpoints AND reconstruction previews both land under the same
        # per-run folder, consistently, rather than two different runs of
        # the same config silently overwriting each other's files. Doesn't
        # break resuming a *specific* crashed run (point
        # resume_from_checkpoint at that run's own timestamped folder);
        # only disables auto-discovering a checkpoint across multiple runs
        # sharing one un-timestamped folder, which trainer.timestamp_run=false
        # opts back into for anyone who specifically wants that instead.
        if bool(t_kwargs.get("timestamp_run", True)):
            run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_dir = str(Path(out_dir) / run_timestamp)
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
        }

        # dataloader_drop_last was never read from config at all -- setting
        # it in a JSON config silently did nothing. HF's own default is
        # False, which means any dataset whose size isn't an exact multiple
        # of the batch size produces a differently-shaped final batch every
        # epoch -- and since this pipeline's pulse layer is jax.jit-compiled
        # (compilation keys on shape), that one odd-sized batch forces a
        # full recompile at every epoch boundary. Default kept at False to
        # match HF's own default and not silently change any existing
        # config's behavior; set trainer.dataloader_drop_last=true to avoid
        # the recompile cost.
        if "dataloader_drop_last" in t_kwargs:
            kwargs["dataloader_drop_last"] = bool(t_kwargs["dataloader_drop_last"])

        if "report_to" in valid_params:
            kwargs["report_to"] = []

        eval_strat = str(t_kwargs.get("eval_strategy", t_kwargs.get("evaluation_strategy", "epoch")))
        if "eval_strategy" in valid_params:
            kwargs["eval_strategy"] = eval_strat
        elif "evaluation_strategy" in valid_params:
            kwargs["evaluation_strategy"] = eval_strat

        # For VAE runs, default to batch_eval_metrics=True: this makes
        # transformers call compute_metrics per eval batch (see
        # IncrementalVAEMetrics) instead of concatenating every batch's
        # reconstructed + target images into one big tensor before calling
        # compute_metrics once. Without this, periodic in-training
        # evaluation on a large validation set (e.g. ImageNet's 50k images)
        # buffers the whole eval set in memory -- a real GPU/host-RAM risk.
        # Config-overridable via training.batch_eval_metrics; not enabled
        # for the classifier path since compute_classification_metrics isn't
        # written to support the per-batch calling convention.
        if parsed.task_type == "vae":
            kwargs["batch_eval_metrics"] = bool(t_kwargs.get("batch_eval_metrics", True))

        dropped = [k for k in kwargs if k not in valid_params]
        if dropped:
            warnings.warn(
                f"build_training_args: dropping kwargs not recognized by the "
                f"installed transformers.TrainingArguments: {dropped}.",
                stacklevel=2,
            )
        final_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return TrainingArguments(**final_kwargs)

    def build_frozen_base_vae(self, base_checkpoint: str, base_model_config_path: str) -> Any:
        """Build + freeze the base VAE a latent_diffusion model decodes
        through, from its OWN config (not the diffusion config) -- so the
        exact architecture used to produce the cached latents is reproduced
        exactly. A mismatch here would silently corrupt every generation
        preview without erroring anywhere."""
        base_cfg = self.load_config(base_model_config_path)
        base_cfg.pop("base_checkpoint", None)
        base_parsed = self.parse(base_cfg)
        base_model = self.build_model(base_parsed)

        base_checkpoint_path = Path(base_checkpoint)
        if not base_checkpoint_path.is_absolute() and not base_checkpoint_path.exists():
            base_checkpoint_path = Path(registered_model_path(base_checkpoint, project_root=self.project_root))
        base_state = torch.load(base_checkpoint_path, map_location="cpu")
        base_model.load_state_dict(base_state, strict=True)
        base_model.eval()
        for p in base_model.parameters():
            p.requires_grad_(False)
        return base_model

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
            perceptual_weight = float(parsed.training_kwargs.get("perceptual_weight", 0.0))
            noise_after_epoch = parsed.training_kwargs.get("noise_after_epoch")
            noise_std = float(parsed.training_kwargs.get("noise_std", 0.1))
            return QuantumVAETrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                kl_weight=kl_weight,
                loss_type=loss_type,
                perceptual_weight=perceptual_weight,
                noise_after_epoch=int(noise_after_epoch) if noise_after_epoch is not None else None,
                noise_std=noise_std,
                image_range=str(parsed.training_kwargs.get("image_range", "0_1")),
                save_reconstructions=bool(parsed.training_kwargs.get("save_reconstructions", True)),
                reconstruction_every_n_epochs=int(parsed.training_kwargs.get("reconstruction_every_n_epochs", 10)),
                reconstruction_num_images=int(parsed.training_kwargs.get("reconstruction_num_images", 8)),
                save_test_reconstructions=bool(parsed.training_kwargs.get("save_test_reconstructions", True)),
                **trainer_kwargs,
            )
        elif parsed.task_type == "latent_diffusion":
            from src.quantum_vae.trainers.latent_diffusion_trainer import LatentDiffusionTrainer
            from src.quantum_vae.trainers.latent_diffusion_data import (
                LatentCacheDataset,
                latent_diffusion_collator,
            )

            variant = str(parsed.model_name).lower()
            if variant == "su2_angles":
                from src.quantum_vae.diffusion.su2_angles import SU2HeatKernelSchedule
                diffusion_schedule = SU2HeatKernelSchedule(
                    n_qubits=int(parsed.model_kwargs.get("n_qubits", 10)),
                    n_timesteps=int(parsed.model_kwargs.get("n_timesteps", 1000)),
                )
            else:
                from src.quantum_vae.diffusion.euclidean import GaussianDiffusionSchedule
                diffusion_schedule = GaussianDiffusionSchedule(
                    n_timesteps=int(parsed.model_kwargs.get("n_timesteps", 1000)),
                )

            cache_path = parsed.data_kwargs["latents_cache"]
            cache_path_resolved = Path(cache_path)
            if not cache_path_resolved.is_absolute():
                cache_path_resolved = self.project_root / cache_path_resolved
            if train_dataset is None:
                train_dataset = LatentCacheDataset(str(cache_path_resolved))
            if eval_dataset is None:
                eval_dataset = train_dataset

            base_vae = self.build_frozen_base_vae(
                parsed.model_kwargs["base_checkpoint"],
                parsed.model_kwargs["base_model_config"],
            )

            return LatentDiffusionTrainer(
                model=model,
                diffusion=diffusion_schedule,
                args=training_args,
                data_collator=latent_diffusion_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                cfg_dropout_prob=float(parsed.training_kwargs.get("cfg_dropout_prob", 0.1)),
                base_vae=base_vae,
                latent_mean=train_dataset.mean,
                latent_std=train_dataset.std,
                latent_shape=tuple(parsed.model_kwargs.get("latent_shape", (4, 7, 7))),
                preview_every_n_epochs=int(parsed.training_kwargs.get("preview_every_n_epochs", 10)),
                preview_digit=int(parsed.training_kwargs.get("preview_digit", 3)),
                preview_num_images=int(parsed.training_kwargs.get("preview_num_images", 8)),
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
) -> Tuple[BaseHFQuantumTrainer, Any, Optional[Dict[str, float]]]:
    """Load config, construct trainer, execute training, evaluate, and save model."""
    trainer = build_trainer_from_config(
        config_source=config_source,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        **kwargs,
    )
    train_output = trainer.train()
    train_results = getattr(train_output, "metrics", train_output)
    eval_results = None
    if eval_dataset is not None or getattr(trainer, "eval_dataset", None) is not None:
        eval_results = trainer.evaluate()
    trainer.save_model()
    return trainer, train_results, eval_results
