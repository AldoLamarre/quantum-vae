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
- `checkpoints/vae/` — VAE run outputs and logs
- `checkpoints/classifier/` — classifier run outputs and logs
- `checkpoints/datasets/` — dataset caches or prepared bundles
- `local_refactor_tests/` — behavior checks

## Main entrypoints

- `python scripts/hf_vae_trainer.py --config <path>`
- `python scripts/hf_classifier_trainer.py --config <path>`
- `python scripts/paper/vaequantumhugface.py`
- `python scripts/paper/vaequantumhugface_cifar.py`
- `python scripts/paper/vaequantumhugface_imagenet.py`
- `python scripts/paper/vaequantumhugface_mnist_pretraining_data_reupload11.py`
- `python scripts/paper/vaequantumhugface_cifar_pretraining_data_reupload.py`

### ⚠️ LPIPS NaN Gradient Issue

LPIPS may produce `NaN` gradients during training due to a numerical instability in its feature normalization when the feature norm is zero.

If you encounter:

```text
RuntimeError: Function 'SqrtBackward0' returned nan values in its 0th output
```

apply the following fix in `lpips/__init__.py` (inside `normalize_tensor()`):

```python
norm_factor = torch.sqrt(
    torch.sum(in_feat**2, dim=1, keepdim=True) + 1e-10
)
```

The `1e-10` epsilon prevents undefined gradients at zero-valued feature vectors.

This is a known issue in the LPIPS implementation:
https://github.com/richzhang/PerceptualSimilarity/issues/121


