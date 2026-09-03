from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from src.quantum_vae.trainers.config_parser import train_from_config


def run_vae_experiment(
    config_path: str | Path,
    train_dataset: Any,
    eval_dataset: Any,
    test_dataset: Optional[Any] = None,
) -> Any:
    trainer, train_results, eval_results = train_from_config(
        config_path,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"Trainer built: {type(trainer).__name__}")
    if train_results is not None:
        print("Training complete.")
        print(train_results)
    if eval_results is not None:
        print("Validation complete.")
        print(eval_results)
    if test_dataset is not None:
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        print("Test complete.")
        print(test_results)

    return trainer
