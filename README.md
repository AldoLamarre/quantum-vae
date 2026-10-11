# Quantum VAE

This repository contains the refactored Quantum VAE experiments and their paper-reproducible launchers.


> **Status:** The repository is currently being refactored. The structure and entrypoints may change before the final release.


## Installation

### NVIDIA (CUDA)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
```

### AMD (ROCm)

```bash
bash setup_remote_rocm.sh
```

Creates `~/venvs/quantum-vae` with ROCm user-space libraries, PyTorch and JAX from AMD's pip index
(`https://stable.repo.amd.com/rocm/whl-next/`), then installs the pinned packages in `requirements-rocm.txt`.
Only the `amdgpu` kernel driver is required on the host. The user must be in the `render` and `video` groups.

**JAX on ROCm 10:** `jaxlib` 0.10.1–0.11.0 does not recognise the `jax_rocm10_plugin` package name, so GPU kernels
fail to load (`No FFI handler registered for hipsolver_*`). The setup script applies AMD's
[`patch_installed_jax_rocm_plugin_names.py`](https://github.com/ROCm/TheRock/tree/main/external-builds/jax) after
installation. The fix ships upstream in `jaxlib` 0.11.1; JAX is kept below 0.11 for PennyLane pulse compatibility.

Use `amd-smi` for monitoring; `rocm-smi` is deprecated.

## Layout

- `src/quantum_vae/` — canonical implementation
- `configs/` — experiment configs
- `configs/paper/` — paper-repro configs
- `scripts/` — generic launchers
- `scripts/paper/` — paper-specific launchers
- `scripts/mnist_ablation/` — archived MNIST ablation scripts
- `checkpoints/vae/` — VAE run outputs and logs
- `checkpoints/classifier/` — classifier run outputs and logs
- `checkpoints/datasets/` — dataset caches or prepared bundles
- `local_refactor_tests/` — behavior checks

## Main entrypoints

- `python scripts/hf_vae_trainer.py --config <path>`
- `python scripts/hf_classifier_trainer.py --config <path>`
- `python scripts/evaluate_vae.py --config <path> --checkpoint <ckpt> [--fid]`
- `python scripts/paper/vaequantumhugface.py`
- `python scripts/paper/vaequantumhugface_cifar.py`
- `python scripts/paper/vaequantumhugface_imagenet.py`
- `python scripts/paper/vaequantumhugface_mnist_pretraining_data_reupload11.py`
- `python scripts/paper/vaequantumhugface_cifar_pretraining_data_reupload.py`

## VAE evaluation notes

- `--checkpoint` accepts legacy PyTorch checkpoint files (such as `.pt`) and newer `.safetensors` files.
- Set `training.image_range` in VAE configs to `"0_1"` or `"-1_1"`.
- Training-time validation reports non-FID metrics (`reconstruction_mse`, `psnr`, `ssim`, `lpips`).
- End-of-training full metrics (including `fid`) are saved to `checkpoints/vae/<run>/evaluation/full_metrics.json`.
