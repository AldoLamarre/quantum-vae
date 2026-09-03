"""Quantum VAE package."""

__all__: list[str] = []

try:
    from .models import (
        AmplitudeClassifierPipeline,
        AnsatzVAEBase,
        ClassifierPipelineConfig,
        PretrainedAnsatzClassifierPipeline,
        QuantumVAEAmplitude,
        QuantumVAEBase,
        QuantumVAEDataReupload,
    )

    __all__.extend(
        [
            "QuantumVAEBase",
            "QuantumVAEAmplitude",
            "QuantumVAEDataReupload",
            "AnsatzVAEBase",
            "ClassifierPipelineConfig",
            "PretrainedAnsatzClassifierPipeline",
            "AmplitudeClassifierPipeline",
        ]
    )
except Exception:
    pass

try:
    from .trainers import (
        BaseHFQuantumTrainer,
        ClassifierDataCollator,
        QuantumClassifierTrainer,
        QuantumVAETrainer,
        TrainerConfigParser,
        VAEDataCollator,
        build_trainer_from_config,
        compute_classification_metrics,
        compute_vae_metrics,
        train_from_config,
    )

    __all__.extend(
        [
            "BaseHFQuantumTrainer",
            "QuantumVAETrainer",
            "QuantumClassifierTrainer",
            "VAEDataCollator",
            "ClassifierDataCollator",
            "compute_classification_metrics",
            "compute_vae_metrics",
            "TrainerConfigParser",
            "build_trainer_from_config",
            "train_from_config",
        ]
    )
except Exception:
    pass
