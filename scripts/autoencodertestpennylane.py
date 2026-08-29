from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from src.quantum_vae.utils.mnist_family import build_mnist_data_bundle
from src.quantum_vae.utils.runtime_utils import resolve_device
import pennylane as qml
from pennylane.templates import QuantumPhaseEstimation
from pennylane import numpy as np
# Get cpu, gpu or mps device for training.
import matplotlib.pyplot as plt
import matplotlib as mpl
torch.autograd.set_detect_anomaly(True)
device = resolve_device()

# Download training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 128
mnist_bundle = build_mnist_data_bundle(training_data, test_data, batch_size=batch_size)
train_dataloader = mnist_bundle["train_dataloader"]
test_dataloader = mnist_bundle["test_dataloader"]

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#fix this for the decorator
#n_estimation_wires = 5
#n_target_wires = 2
#dev = qml.device("default.qubit", wires=n_estimation_wires + 2)
#dev = qml.device("lightning.qubit", wires=n_estimation_wires + n_target_wires)
#target_wires = [0, 1]
#target_wires = list(range(n_target_wires))
#eigenvector_param_size = 2 ** n_target_wires
#unitary_param_size = 4 ** n_target_wires - 1


#@qml.batch_input(argnum=1)
#, diff_method="backprop", interface="torch"
#@qml.qnode(dev)
# def circuitl(inputs):
#     estimation_wires = range(n_target_wires, n_estimation_wires + n_target_wires)
#     #print("inputs = " + str(inputs))
#     eigenvector, unitary = torch.split(inputs, [eigenvector_param_size,unitary_param_size])
#     unitary = qml.ArbitraryUnitary(weights=unitary, wires=target_wires)
#     qml.QubitStateVector(torch.sqrt(eigenvector), wires=target_wires )
#     QuantumPhaseEstimation(
#         unitary,
#         estimation_wires=estimation_wires,
#     )
#     return qml.probs(estimation_wires)

#weight_shapes = {}
#qlayer = qml.qnn.TorchLayer(circuitl, weight_shapes)
class NeuralNetwork(nn.Module):
    def __init__(self, wires=8):
        super().__init__()
        self.flatten = nn.Flatten()
        #self.n_estimation_wires = 3
        #self.n_target_wires = 2
        self.wires = np.arange(wires)
        self.dev = qml.device("default.qubit", wires = self.wires)
        #self.n_estimation_wires = 5
        #self.target_wires = [0, 1]
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

        )
        self.real = nn.Sequential(
            nn.Linear(256, 2 ** len(self.wires)),

        )
        self.img = nn.Sequential(
            nn.Linear(256, 2 ** len(self.wires)),

        )


        self.decoder = nn.Sequential(
            nn.Linear(2**len(self.wires), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),
        )
        #weight_shapes = {}
        #self.qlayer = qml.qnn.TorchLayer(self.circuitl, v)
        self.qlayer = qml.qnn.TorchLayer(self.contruct_circuit(), weight_shapes={})

    def forward(self, x):
        x = self.flatten(x)
        encodetemp = self.encoder(x)
        real = self.real(encodetemp)
        img = self.img(encodetemp)
        state = torch.complex(real, img)
        encode = torch.nn.functional.normalize(state, dim=1)
        #print(encode)
        #weight_shapes =#{}

        #print(inputs)

        quantumlatent = self.qlayer(encode)
        #print(quantumlatent)
        decode = self.decoder(torch.abs(quantumlatent))
        #print(phase_estimated)
        #print(finalinputs.shape)

        return decode.view(-1, 1, 28, 28)
    def get_latent(self, x):
        x = self.flatten(x)
        encodetemp = self.encoder(x)
        real = self.real(encodetemp)
        img = self.real(encodetemp)
        state = torch.complex(real, img)
        encode = torch.nn.functional.normalize(state, dim=1)
        # print(encode)
        # weight_shapes =#{}

        # print(inputs)

        #quantumlatent = self.qlayer(encode)
        return encode

    def contruct_circuit(self):
        #@qml.batch_input(argnum=2)
        @qml.qnode(self.dev, diff_method="backprop", interface="torch", wires=self.wires)
        def circuit(inputs):
            #print(torch.sum(inputs,dim=1))
            #qml.QubitStateVector(torch.sqrt(inputs.type(torch.torch.complex128)), wires=self.wires)
            #qml.QubitStateVector(inputs.type(torch.torch.complex128), wires=self.wires)
            qml.QubitStateVector(inputs, wires=self.wires)

            return qml.probs(wires=self.wires)
        return circuit



model = NeuralNetwork(wires=10).to(device)
print(model)
loss_fn = nn.BCELoss(reduction='sum')
#loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
def getnoise(inputs):
    noise = torch.zeros(inputs.size(0), 1, 28, 28)
    nn.init.normal_(noise, 0, 0.1)
    noise = noise.to(device)
    noise_inputs = inputs + noise
    return noise_inputs

def train(dataloader, model, loss_fn, optimizer, noise=False):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X =  X.to(device)
        if noise:
            inputs = getnoise(X)
        else:
            inputs = X
        # zero the parameter gradients
        optimizer.zero_grad()
        # Compute prediction error
        pred = model(inputs)
        loss = loss_fn(pred, X)

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn,noise=False):
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
            pred = model(inputs)
            test_loss += loss_fn(pred, X).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        #correct /= size
        print(f"testloss: {test_loss:>7f}")
        out_img = torch.squeeze(pred.cpu().data)
        print(out_img.size())

    for i in range(5):
        plt.subplot(1, 3, 1)
        plt.imshow(torch.squeeze(inputs[i]).cpu().numpy(), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(torch.squeeze(X[i]).cpu().numpy(), cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(out_img[i].numpy(), cmap='gray')
        state = model.get_latent(inputs[i]).cpu().detach().numpy()
        #print(str(model.get_latent(inputs[i]).cpu().detach().numpy()))
        #string = str(model.get_latent(inputs[i]).cpu().detach().numpy())
        #txt = 'Sate Vector : ' + string
        fidelity = np.empty((1,5))
        for j in range(5):
            print(np.square(np.abs(np.dot(state, model.get_latent(inputs[j]).cpu().detach().numpy().transpose()))))

        #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        #plt.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        plt.show()
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

epochs = 25
for t in range(epochs):
    print(f"Epoch Noise {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,noise=True)
    test(test_dataloader, model, loss_fn,noise=True)
print("Done!")
