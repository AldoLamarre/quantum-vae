from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.quantum_vae.utils.hf_classifier_config import main


def _cli_entry(config_path: str | Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train a config-driven HF classifier experiment.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file")
    args = parser.parse_args()

    selected = Path(config_path) if config_path is not None else Path(args.config)
    if not selected.is_absolute():
        selected = ROOT / selected
    main(selected)


if __name__ == "__main__":
    _cli_entry()
