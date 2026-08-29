from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quantum_vae.utils.hf_cifar_classifier_config import main


if __name__ == "__main__":
    main()
