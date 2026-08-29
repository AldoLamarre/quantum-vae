from typing import Optional, Union

import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from torch import nn
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pennylane as qml
from pennylane.templates import QuantumPhaseEstimation
from pennylane import numpy as np
# Get cpu, gpu or mps device for training.
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkblocks as nb

torch.autograd.set_detect_anomaly(True)
import os
import datetime
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
path = "log/Benchmarking/linearSeprable/vae kl/"+date+"/"
os.makedirs(path)
torch.autograd.set_detect_anomaly(True)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context()



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class MLPVAE(nn.Module):
    def __init__(self, wires=8, dimention=2, hiddensize=512):
        super().__init__()
        #self.flatten = nn.Flatten()
        #self.n_estimation_wires = 3
        #self.n_target_wires = 2
        self.wires = np.arange(wires)
        self.dev = qml.device("default.qubit", wires = self.wires)
        #self.n_estimation_wires = 5
        #self.target_wires = [0, 1]
        self.encoder = nn.Sequential(
            nn.Linear(dimention*dimention, hiddensize),
            nn.LayerNorm(hiddensize),
            nn.ReLU(),
            nn.Linear(hiddensize, hiddensize//2),
            nn.LayerNorm(hiddensize//2),
            nn.ReLU(),

        )
        self.realmu = nn.Sequential(
            nn.Linear(hiddensize, 2 ** len(self.wires)),

        )
        self.realsigma = nn.Sequential(
            nn.Linear(hiddensize, 2 ** len(self.wires)),

        )
        self.imgmu = nn.Sequential(
            nn.Linear(hiddensize, 2 ** len(self.wires)),

        )
        self.imgsigma = nn.Sequential(
            nn.Linear(hiddensize, 2 ** len(self.wires)),

        )
        self.normale = torch.distributions.Normal(0, 1)

        self.lk=0
        self.klvar = 0, 0, 0, 0


        self.decoder = nn.Sequential(
            nn.Linear(2**len(self.wires), hiddensize//2),
            nn.LayerNorm(hiddensize//2),
            nn.ReLU(),
            nn.Linear(hiddensize//2, hiddensize),
            nn.LayerNorm(hiddensize),
            nn.ReLU(),
            nn.Linear(hiddensize, dimention),
            nn.Sigmoid(),
        )
        #weight_shapes = {}
        #self.qlayer = qml.qnn.TorchLayer(self.circuitl, v)
        self.qlayer = qml.qnn.TorchLayer(self.contruct_circuit(), weight_shapes={})

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z
    def forward(self, x):
        encode = self.get_latent(x)
        #print(encode)
        #weight_shapes =#{}

        #print(inputs)

        quantumlatent = self.qlayer(encode)
        #print(quantumlatent)
        decode = self.decoder(torch.abs(quantumlatent))
        #print(phase_estimated)
        #print(finalinputs.shape)

        return decode,quantumlatent
    def get_latent(self, x):
        x = self.flatten(x)
        encodetemp = self.encoder(x)
        realmu = self.realmu(encodetemp)
        realsigma = self.realsigma(encodetemp)
        imgmu = self.imgmu(encodetemp)
        imgsigma = self.imgsigma(encodetemp)
        real = self.reparameterization(realmu, torch.exp(0.5 * realsigma))
        img = self.reparameterization(imgmu, torch.exp(0.5 * imgsigma))
        state = torch.complex(real, img)

        encode = torch.nn.functional.normalize(state, dim=1)
        self.klvar = realmu, realsigma, imgmu, imgsigma

        self.kl = - 0.5 * (torch.sum(1 + realsigma - realmu.pow(2) - realsigma.exp()) +
                           torch.sum(1 + imgsigma - imgmu.pow(2) - imgsigma.exp()))
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
            #for i in self.wires:
                #qml.BitFlip(0.21, i)
            return qml.probs(wires=self.wires)
        return circuit

def train(dataloader, model, loss_fn, optimizer, epoch=0):
    size =len(inputs)
    inputs, labels = dataloader
    model.train()
    total_loss = 0.0
    batch=0
    #for (X, y) in range(dataloader["inputs"],dataloader["labels"]):
    inputs =  inputs.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()
    # Compute prediction error
    pred,quantumstate = model(inputs)
    #statecycle = model.get_latent(pred)

    #fidelity = torch.square(torch.abs(torch.matmul(quantumstate, torch.t(statecycle))))
    # cycleloss = torch.mean(torch.sqrt(1.0-fidelity))
    # cycleloss = -torch.mean(fidelity)
    #cycleloss = torch.mean(1.0 - fidelity)
    loss = loss_fn(pred, inputs) + 0.0001*model.kl #+ 0.0001*cycleloss
    total_loss += loss.item()

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1)

    #if batch == 100:
        #nb.plot_grad_flow(path,model.named_parameters(),"VAE", epoch)

    optimizer.step()
    optimizer.zero_grad()
    #if batch % 100 == 0:
    loss, current = loss.item(), (batch + 1) * len(X)
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
def test(dataloader, model, loss_fn,epoch=0):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            inputs = X
            pred, state = model(inputs)
            test_loss += loss_fn(pred, X).item() + model.kl
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        print(f"testloss: {test_loss:>7f}")

    return test_loss

if __name__ == '__main__':
    dimention = 20
    model = MLPVAE(wires=1,dimention=dimention,hiddensize=8).to(device)
    print(model)
    datasets = qml.data.Dataset.open(filepath="datasets/other/LinearlySeparable/LinearlySeparable_LinearlySeparable.h5")
    inputs_matrix = [input for input in datasets.train[str(dimention)]["inputs"]]
    labels_matrix = [label for label in datasets.train[str(dimention)]["labels"]]
    dataloader=(inputs_matrix,labels_matrix)
    print(inputs_matrix)

    # loss_fn = nn.BCELoss(reduction='sum')
    loss_val = 1000000
    loss_fn = nn.MSELoss()
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(datasets.train[str(dimention)], model, loss_fn, optimizer, epoch=t)
        #loss = test(val_dataloader, model, loss_fn, epoch=t)
        #if loss < loss_val:
            #loss_val = loss
            #torch.save(model.state_dict(), "variationalautoencodertestpennylane new five.pt")
    #test(test_dataloader, model, loss_fn, noise=True, epoch=epochs+1)
    print("Done!")