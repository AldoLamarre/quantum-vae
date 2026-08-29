import datetime
import os
from pathlib import Path

import torch


def make_timestamp():
    return datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")


def create_run_path(base_dir, timestamp=None, include_timestamp=True):
    path = Path(base_dir)
    if include_timestamp:
        path = path / (timestamp or make_timestamp())
    path.mkdir(parents=True, exist_ok=True)
    return f"{path.as_posix()}/"


def configure_cuda_visible_devices(device_id, only_if_unset=False):
    if only_if_unset and os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def resolve_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

