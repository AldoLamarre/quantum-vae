from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from taming.modules.losses.lpips import LPIPS
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

from src.quantum_vae.models import QuantumVAEAmplitude


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "hf_cifar10_transfer_vae.json"
)


def resolve_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("tomography", False)
    cfg["tomography"] = False
    return cfg


def getnoise(inputs: torch.Tensor, resolution: int, device: str) -> torch.Tensor:
    noise = torch.zeros(inputs.size(0), 3, resolution, resolution)
    nn.init.normal_(noise, 0, 0.1)
    noise = noise.to(device)
    return inputs + noise


def build_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_cfg = config["data"]
    transform = Compose([Resize((data_cfg["resolution"], data_cfg["resolution"])), ToTensor()])
    training_data = datasets.CIFAR10(
        root=data_cfg["root"],
        train=True,
        download=True,
        transform=transform,
    )
    test_data = datasets.CIFAR10(
        root=data_cfg["root"],
        train=False,
        download=True,
        transform=transform,
    )

    train_size = int(data_cfg["train_size"])
    val_size = int(data_cfg["val_size"])
    train_set, val_set = torch.utils.data.random_split(training_data, [train_size, val_size])
    batch_size = int(data_cfg["batch_size"])
    return (
        DataLoader(train_set, batch_size=batch_size),
        DataLoader(val_set, batch_size=batch_size),
        DataLoader(test_data, batch_size=batch_size),
    )


def train_epoch(dataloader, model, loss_fn, optimizer, device, training_cfg, noise=False, epoch=0):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    for batch, (X, _y) in enumerate(dataloader):
        X = X.to(device)
        inputs = getnoise(X, int(training_cfg["resolution"]), device) if noise else X
        optimizer.zero_grad()
        output, kl, _states = model.forward_transfer(inputs, tomography=False)
        pred = output.sample
        loss = torch.mean(loss_fn(pred.contiguous(), X.contiguous())) + float(training_cfg["kl_weight"]) * kl
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Average loss for the batch: {total_loss/size}")


def eval_epoch(dataloader, model, loss_fn, device, training_cfg, output_dir: Path, noise=False, epoch=0):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, _y in dataloader:
            X = X.to(device)
            inputs = getnoise(X, int(training_cfg["resolution"]), device) if noise else X
            output, kl, _states = model.forward_transfer(inputs, tomography=False)
            pred = output.sample
            test_loss += torch.mean(loss_fn(pred.contiguous(), X.contiguous())) + float(training_cfg["kl_weight"]) * kl
        test_loss /= num_batches
        print(f"testloss: {test_loss:>7f}")
        out_img = torch.squeeze(pred.cpu().data)

    fig, axs = plt.subplots(10, 3, figsize=(10, 10))
    for i in range(10):
        axs[i, 0].imshow(torch.squeeze(inputs[i].cpu().detach()).numpy().transpose(1, 2, 0))
        axs[i, 1].axis("off")
        axs[i, 1].imshow(torch.squeeze(X[i].cpu().detach()).numpy().transpose(1, 2, 0))
        axs[i, 1].axis("off")
        axs[i, 2].imshow(out_img[i].cpu().detach().numpy().transpose(1, 2, 0))
        axs[i, 2].axis("off")
    plt.savefig(output_dir / f"Epoch {epoch}.png")
    plt.close(fig)
    return test_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device()
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))

    train_loader, val_loader, test_loader = build_dataloaders(config)

    model_cfg = config["model"]
    model = QuantumVAEAmplitude(
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        sample_size=model_cfg["sample_size"],
        block_out_channels=tuple(model_cfg["block_out_channels"]),
        down_block_types=tuple(model_cfg["down_block_types"]),
        up_block_types=tuple(model_cfg["up_block_types"]),
    ).to(device)

    training_cfg = config["training"]
    loss_fn = LPIPS().to(device).eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg["learning_rate"]))

    output_root = Path(config["output"]["root"])
    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    output_dir = output_root / "Prob" / date
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(config["base_checkpoint"])
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    print("model loaded")

    best_loss = float("inf")
    best_path = output_dir / str(config["output"]["best_checkpoint_name"])
    epochs = int(training_cfg["epochs"])
    noise_after_epoch = int(training_cfg["noise_after_epoch"])

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            device,
            training_cfg,
            noise=(t > noise_after_epoch),
            epoch=t,
        )
        loss = eval_epoch(val_loader, model, loss_fn, device, training_cfg, output_dir, epoch=t)
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=torch.device(device)))
    eval_epoch(
        test_loader,
        model,
        loss_fn,
        device,
        training_cfg,
        output_dir,
        noise=True,
        epoch=epochs + 1,
    )
    print("Done!")


if __name__ == "__main__":
    main()
