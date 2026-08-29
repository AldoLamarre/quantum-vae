from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import datetime
import os
from src.quantum_vae.utils.cifar_family import build_cifar10_data_bundle
from src.quantum_vae.utils.runtime_utils import create_run_path, resolve_device

date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
path = create_run_path("log/classicalddpm/cifar10/hugfaceddpm", timestamp=date)
os.makedirs(path, exist_ok=True)

torch.autograd.set_detect_anomaly(True)
device = resolve_device()

# Download training data from open datasets.
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)