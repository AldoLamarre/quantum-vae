# Quantum VAE

This repository contains the refactored Quantum VAE experiments and their paper-reproducible launchers.


> **Status:** The repository is currently being refactored. The structure and entrypoints may change before the final release.


## Layout

- `src/quantum_vae/` — canonical implementation
- `configs/` — experiment configs
- `configs/paper/` — paper-repro configs
- `scripts/` — generic launchers
- `scripts/paper/` — paper-specific launchers
- `scripts/mnist_ablation/` — archived MNIST ablation scripts
- `local_refactor_tests/` — behavior checks

## Main entrypoints

- `python scripts/hf_vae_trainer.py --config <path>`
- `python scripts/hf_classifier_trainer.py --config <path>`
- `python scripts/paper/vaequantumhugface.py`
- `python scripts/paper/vaequantumhugface_cifar.py`
- `python scripts/paper/vaequantumhugface_imagenet.py`
- `python scripts/paper/vaequantumhugface_mnist_pretraining_data_reupload11.py`
- `python scripts/paper/vaequantumhugface_cifar_pretraining_data_reupload.py`

