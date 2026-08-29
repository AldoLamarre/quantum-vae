from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = "models"
CHECKPOINTS_DIR = "checkpoints"

REGISTERED_MODELS = {
    "mnist_pennylane_vae_five": {
        "managed": "models/autoencoders/mnist/variationalautoencodertestpennylane_five.pt",
        "legacy": ["variationalautoencodertestpennylane five.pt"],
    },
    "mnist_pennylane_vae_new_five": {
        "managed": "models/autoencoders/mnist/variationalautoencodertestpennylane_new_five.pt",
        "legacy": ["variationalautoencodertestpennylane new five.pt"],
    },
    "hf_mnist_autoencoderkl": {
        "managed": "models/autoencoders/mnist/autoencoderkl.pt",
        "legacy": ["autoencoderkl.pt"],
    },
    "hf_cifar_autoencoderkl": {
        "managed": "models/autoencoders/cifar/autoencoderkl.pt",
        "legacy": ["autoencoderkl.pt"],
    },
    "hf_imagenet_autoencoderkl": {
        "managed": "models/autoencoders/imagenet/autoencoderkl.pt",
        "legacy": ["autoencoderkl.pt"],
    },
    "cifar10_autoencoderkl": {
        "managed": "models/autoencoders/cifar10/autoencoderklcifar10.pt",
        "legacy": ["autoencoderklcifar10.pt", "results/autoencoderklcifar10.pt"],
    },
}


def _project_root(project_root=None):
    return Path(project_root) if project_root is not None else PROJECT_ROOT


def models_dir(project_root=None, create=False):
    path = _project_root(project_root) / MODELS_DIR
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoints_dir(project_root=None, create=False):
    path = _project_root(project_root) / CHECKPOINTS_DIR
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def managed_model_path(model_key, project_root=None, create_parent=False):
    model_spec = REGISTERED_MODELS[model_key]
    path = _project_root(project_root) / model_spec["managed"]
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def registered_model_path(model_key, project_root=None):
    model_spec = REGISTERED_MODELS[model_key]
    root = _project_root(project_root)
    managed_path = root / model_spec["managed"]
    if managed_path.exists():
        return str(managed_path)

    legacy_paths = [root / relative_path for relative_path in model_spec["legacy"]]
    for legacy_path in legacy_paths:
        if legacy_path.exists():
            return str(legacy_path)

    return str(legacy_paths[0])

