from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import datetime
import os
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
path = "log/classicalddpm/cifar10/hugfaceddpm/"+date+"/"
os.makedirs(path)

torch.autograd.set_detect_anomaly(True)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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