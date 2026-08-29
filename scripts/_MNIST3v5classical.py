from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from model_paths import registered_model_path
from runtime_utils import create_run_path, resolve_device
from src.quantum_vae.utils.mnist_3v5_family import build_mnist_3v5_dataset_bundle

import numpy as np
import math
torch.autograd.set_detect_anomaly(True)
path = create_run_path("paperlogs/3v5CC_Classicalpost")
device = resolve_device()

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=False,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor(),
)


batch_size = 128
mnist_bundle = build_mnist_3v5_dataset_bundle(
    training_data,
    test_data,
    batch_size=batch_size,
)
filtered_train = mnist_bundle["filtered_train"]
filtered_test = mnist_bundle["filtered_test"]
train_set = mnist_bundle["train_set"]
val_set = mnist_bundle["val_set"]
train_dataloader = mnist_bundle["train_dataloader"]
val_dataloader = mnist_bundle["val_dataloader"]
test_dataloader = mnist_bundle["test_dataloader"]

test_set = filtered_test

# Preserve the current script semantics while sharing the common MNIST 3-vs-5 setup.


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(512),
            nn.BatchNorm1d(512),
            nn.Linear(10, 2),

        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.model(x)
def train(dataloader, model,model_vae,loss_fn, optimizer, epoch=0):
    size = len(dataloader.dataset)
    model.train()
    epoch += 1
    model_vae.eval()
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X = X.view(-1,1,28,28).to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            state = model_vae.get_latent(X)

        #copies = torch.einsum("bi,bj->bij",state,state).view(-1,state.shape[-1]*state.shape[-1])

        #out = model(torch.cat(((state.real),(state.imag))))
        out= model(X.view(-1,784))
        #print(out.grad_fn)

        loss = loss_fn(out, y.to(device))

        total_loss += loss.item()

        # Backpropagation

        #for param in model.parameters():
            #if param.grad is None:
                #print("No gradient for parameter:", param)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model,model_vae,loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    model_vae.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            #state = model_vae.get_latent(X)

            #copies = torch.einsum("bi,bj->bij", state, state).view(-1, state.shape[-1] * state.shape[-1])

            #pred = model(torch.stack([torch.real(state), torch.imag(state)], dim=1))
            pred = model(X.view(-1, 784))
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


if __name__ == '__main__':
    qubit = 10
    from variationalautoencodertestpennylane import NeuralNetwork

    modelVAE = NeuralNetwork(wires=qubit/2).to(device)
    modelVAE.load_state_dict(
        torch.load(registered_model_path("mnist_pennylane_vae_five"), map_location=device)
    )

    loss_fn = nn.CrossEntropyLoss()

    model = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epochs = 500

    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    best_val_Epoch = float('inf')  # Initialize best validation loss as infinity
    writer = SummaryWriter(path+'loss_logging')  # TensorBoard writer


    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, modelVAE,loss_fn, optimizer, t)
        val_loss, val_accuracy = test(val_dataloader, model, modelVAE, loss_fn)

        # Log validation loss to TensorBoard
        writer.add_scalar('Validation Loss', val_loss, t)
        writer.add_scalar('Validation Acurracy', val_accuracy, t)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_accuracy = val_accuracy
            best_val_Epoch = t
            torch.save(model.state_dict(), path + "best_model.pth")
            print(f"New best validation loss: {best_val_loss:.4f} acurracy: {best_val_loss_accuracy:.4f} - Model saved.")

        # After training, run the test function with test data

    model.load_state_dict(torch.load(path + "best_model.pth"))

    test_loss, test_accuracy = test(test_dataloader, model, modelVAE, loss_fn)
    writer.add_scalar('Test Loss', test_loss, best_val_Epoch)
    writer.add_scalar('Test Accuracy', test_accuracy, best_val_Epoch)
    print(f"Test Loss after training: {test_loss:.4f} acurracy: {test_accuracy:.4f}")