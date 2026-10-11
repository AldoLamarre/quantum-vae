#!/usr/bin/env bash
# Create the project venv on an AMD/ROCm host (torch + jax on GPU).
# ROCm user-space libraries are installed into the venv from AMD's pip index,
# so only the amdgpu kernel driver is required on the host.
# Usage: bash setup_remote_rocm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -z "${PROJECT_DIR:-}" ]]; then
  for d in "$SCRIPT_DIR" "$SCRIPT_DIR/.."; do
    [[ -f "$d/requirements-rocm.txt" ]] && PROJECT_DIR="$(cd "$d" && pwd)" && break
  done
fi
: "${PROJECT_DIR:?requirements-rocm.txt not found; set PROJECT_DIR}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/quantum-vae}"
PYTHON="${PYTHON:-python3.12}"
GFX="${GFX:-gfx1100}"
ROCM_INDEX="${ROCM_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}"
TORCH_VERSION="${TORCH_VERSION:-2.13.*}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.28.*}"
JAX_VERSION="${JAX_VERSION:-0.10.2}"
ROCM_MAJOR="${ROCM_MAJOR:-10}"
JAX_PATCH_URL="https://raw.githubusercontent.com/ROCm/TheRock/main/external-builds/jax/patch_installed_jax_rocm_plugin_names.py"

"$PYTHON" -m venv "$VENV_DIR"
PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"
"$PIP" install --upgrade pip wheel

"$PIP" install --index-url "$ROCM_INDEX" "rocm[libraries,device-${GFX}]"
"$PIP" install --index-url "$ROCM_INDEX" \
    "torch[device-${GFX}]==${TORCH_VERSION}" "torchvision[device-${GFX}]==${TORCHVISION_VERSION}"

"$PIP" install --index-url "$ROCM_INDEX" \
    "jax_rocm${ROCM_MAJOR}_plugin==${JAX_VERSION}.*" "jax_rocm${ROCM_MAJOR}_pjrt==${JAX_VERSION}.*"
"$PIP" install "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"

# jaxlib < 0.11.1 does not recognise the jax_rocm10 plugin name; patch in place (no-op otherwise).
PATCH="$(mktemp --suffix=.py)"
curl -fsSL "$JAX_PATCH_URL" -o "$PATCH"
"$PY" "$PATCH" --plugin-package "jax_rocm${ROCM_MAJOR}_plugin"
rm -f "$PATCH"

"$PIP" install -r "$PROJECT_DIR/requirements-rocm.txt"
"$PIP" install -e "$PROJECT_DIR" --no-deps

"$PY" - <<'EOF'
import torch, jax, jax.numpy as jnp
print("torch", torch.__version__, "| hip", torch.version.hip)
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i}", torch.cuda.get_device_name(i))
print("jax", jax.__version__, "| backend:", jax.default_backend(), "| devices:", jax.devices())
x = jnp.ones((1024, 1024))
print("  jax matmul ok:", float((x @ x).sum()))
EOF
