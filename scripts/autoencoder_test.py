from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Optional, Union

from pathlib import Path

from taming.modules.losses.lpips import LPIPS
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.tensorboard import SummaryWriter
from model_paths import registered_model_path

import matplotlib.pyplot as plt

## torch.autograd.set_detect_anomaly(True)
import os
import datetime
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
#path =  "paperlogs/CIFAR10/CC_classicpost-c10_replay/"+date+"/"



# encoder_name = "imagenet"
# dataset_name = 'ImageNet'

#encoder_name = "cifar10-transfer"
encoder_name = "cifar10"
dataset_name = "cifar10"

resolution = 32 # 32 # 256
batch_size = 32
num_channels = 3 # RGB

path = Path("paperlogs/autoencoder_test/") / encoder_name / dataset_name / date

# TODO: move when creating files
path.mkdir(parents=True)

from reconstruction_metrics import MetricsTracker

#device = torch.device("cpu")
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

torch.manual_seed(42)
#if device.type == 'cuda':
torch.cuda.manual_seed_all(42)

print("device =", device, " device count ", torch.cuda.device_count())

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_dataset(name, transform, root="./data", train=True):
    name = name.lower()

    if name == "cifar10":
        return datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    elif name == "cifar100":
        return datasets.CIFAR100(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    elif name == "imagenet":
        split = "train" if train else "val"
        return datasets.ImageNet(
            root=root,
            split=split,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

transform = Compose([Resize((resolution,resolution)), ToTensor()])
test_data = Subset(get_dataset(dataset_name, transform, train=False), range(128))

# Download training data from open datasets.
# training_data = DatasetClass(
#     root="data",
#     train=True,
#     download=True,
#     transform=Compose([Resize((resolution,resolution)),ToTensor()]), #ToTensor(),
# )

# Download test data from open datasets.
# test_data = DatasetClass(
#     root="data",
#     split="val",
#     #train=False,
#     # download=True,
#     transform=Compose([Resize((resolution,resolution)),ToTensor()]), #ToTensor(),
# )


# print(len(training_data))
# print(len(training_data.classes))
#
# train_set, val_set = torch.utils.data.random_split(training_data, [40000, 10000])

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size)

import csv

class ProgressiveCSV:
    def __init__(self, filename, headers):
        # Open the file and prepare the CSV writer
        self._file = open(filename, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=headers)
        self._writer.writeheader()

    def add_line(self, row_dict):
        # Write a single row (keys should match headers)
        self._writer.writerow(row_dict)
        self._file.flush()  # ensure it’s written to disk

    def close(self):
        self._file.close()

def getnoise(inputs):
    noise = torch.zeros(inputs.size(0), 3, resolution, resolution)
    nn.init.normal_(noise, 0, 0.1)
    noise = noise.to(device)
    noise_inputs = inputs + noise
    return noise_inputs

def test(dataloader, model, loss_fn, noise=False, epoch=0):
    csv_path = path/"metrics.csv"
    per_image_csv = path/"per_image.csv"
    metrics = MetricsTracker(device, csv_path=str(csv_path), per_image_csv=str(per_image_csv), compute_fid=True)
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            #X, y = X.to(device), y.to(device)
            X = X.to(device)
            if noise:
                inputs = getnoise(X)
            else:
                inputs = X
            out, kl, states = model(inputs)
            #print(states.shape)
            pred = out.sample
            test_loss +=  torch.mean(loss_fn(pred.contiguous(), X.contiguous())) + 0.00000001 * kl.mean()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            # pass loss_fn and X so per-image loss (no KL) is recorded
            metrics.update_and_log(X, pred, loss_fn=loss_fn)

        test_loss /= num_batches

        #correct /= size
        print(f"testloss: {test_loss:>7f}")
        out_img = torch.squeeze(pred.cpu().data)
        print(out_img.size())

    results = metrics.finalize(epoch)

    # TODO: Recuperer les 3 métriques de ~/pennylanetest/sample/compute_reconstruction_metrics.py

    def create_image(image_name: Path, image: torch.Tensor):
        fig, axs = plt.subplots(1, 1)
        axs.imshow(image)
        axs.axis('off')
        plt.savefig(image_name.absolute())
        plt.show()
        plt.close(fig)

    # fig, axs = plt.subplots(10, 3, figsize=(10, 10))  # Create a figure and a set of subplots
    for i in range(10):
        inputs_image = torch.squeeze(inputs[i].cpu().detach()).numpy().transpose(1,2,0)
        create_image(path / f'Epoch-{epoch}-X{i}.png', inputs_image)

        if noise:
            X_image = torch.squeeze(X[i].cpu().detach()).numpy().transpose(1,2,0)
            create_image(path / f'Epoch-{epoch}-noisy{i}.png', X_image)

        X_reconstruction_image = out_img[i].cpu().detach().numpy().transpose(1,2,0)
        create_image(path / f'Epoch-{epoch}-reconstruction{i}.png', X_reconstruction_image)


    #     axs[i, 0].imshow(torch.squeeze(inputs[i].cpu().detach()).numpy().transpose(1, 2, 0))
    #     axs[i, 0].axis('off')
    #     axs[i, 1].imshow(torch.squeeze(X[i].cpu().detach()).numpy().transpose(1, 2, 0))
    #     axs[i, 1].axis('off')
    #     axs[i, 2].imshow(out_img[i].cpu().detach().numpy().transpose(1, 2, 0))
    #     axs[i, 2].axis('off')
        # state = model.get_latent(inputs[i]).cpu().detach().numpy()
        # print(str(model.get_latent(inputs[i]).cpu().detach().numpy()))
        # string = str(model.get_latent(inputs[i]).cpu().detach().numpy())
        # txt = 'Sate Vector : ' + string
        # fidelity = np.empty((1,5))
        # for j in range(10):
        # print("Fidelity a "+ str(y[i]) + " vs  " + str(y[j])+ " " + str(np.square(np.abs(np.dot(state, model.get_latent(inputs[j]).cpu().detach().numpy().transpose())))))

        # plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        # plt.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        #
    # plt.savefig(path + "Epoch " + str(epoch) + ".png")
    # plt.show()
    # plt.close(fig)

    return test_loss

def test_imagenet():
    pass


def test_cifar10():
    qubit = 7 # -> so 'state' is 2048
    #from variationalautoencodertestpennylane import NeuralNetwork
    from  vaequantumhugface_cifar import quantumautoencoder
    # from vaequantumhugface_cifar10_transfer import quantumautoencoder
    # from vaequantumhugface_imagenet_claude_nonsteaming import QuantumAutoencoder as quantumautoencoder
    modelVAE = quantumautoencoder(
        in_channels=num_channels,
        out_channels=num_channels,
        sample_size=resolution, # chamge later to 128
        block_out_channels=(32, 32, 64),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    ).to(device)
    modelVAE.set_Tomo(False)
    print(modelVAE) ##  NeuralNetwork(wires=qubit/2).to(device)
    #modelVAE = torch.compile(modelVAE)
    print("device =", device, " device count ", torch.cuda.device_count())
    # modelVAE.load_state_dict(torch.load("autoencoderkl_imagenet-epoch-1.pt", map_location=device))
    state_dict = torch.load(registered_model_path("cifar10_autoencoderkl"), map_location=device)
    modelVAE.load_state_dict(state_dict)
    loss_fn = LPIPS().to(device).eval()
    test(test_dataloader, modelVAE, loss_fn)


if __name__ == '__main__':
    test_cifar10()
