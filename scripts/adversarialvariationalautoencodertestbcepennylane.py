from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import pennylane as qml
from pennylane.templates import QuantumPhaseEstimation
from pennylane import numpy as np
# Get cpu, gpu or mps device for training.
import matplotlib.pyplot as plt
import matplotlib as mpl
torch.autograd.set_detect_anomaly(True)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

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
        self.realmu = nn.Sequential(
            nn.Linear(256, 2 ** len(self.wires)),

        )
        self.realsigma = nn.Sequential(
            nn.Linear(256, 2 ** len(self.wires)),

        )
        self.imgmu = nn.Sequential(
            nn.Linear(256, 2 ** len(self.wires)),

        )
        self.imgsigma = nn.Sequential(
            nn.Linear(256, 2 ** len(self.wires)),

        )
        self.normale = torch.distributions.Normal(0, 1)

        self.lk=0
        self.klvar = 0, 0, 0, 0


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

        return decode.view(-1, 1, 28, 28)
    def get_latent(self, x):
        x = self.flatten(x)
        encodetemp = self.encoder(x)
        realmu = self.realmu(encodetemp)
        realsigma = self.realsigma(encodetemp)
        imgmu = self.imgmu(encodetemp)
        imgsigma = self.imgmu(encodetemp)
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

#From Ai505
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            ############################
            # Define your own discriminator #
            ############################
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()

            ############################
        )

    def forward(self, input):
        #####################################
        # Change the shape of output if necessary #

        #####################################
        input = input.view(-1, 28 * 28)
        output = self.main(input)

        #####################################
        # Change the shape of output if necessary #

        #####################################
        output = output.squeeze(dim=1)

        return output

model = NeuralNetwork(wires=4).to(device)
modelD = Discriminator().to(device)
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.relu(1. - logits_real))
    loss_fake = torch.mean(torch.nn.functional.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
print(model)
lossD_fn = nn.BCELoss()
lossG_fn = nn.BCELoss()
loss_fn = nn.MSELoss()
#loss_fn = nn.BCELoss(reduction="sum")
#loss_kl = nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizerD = torch.optim.AdamW(modelD.parameters(), lr=1e-5)
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
        # Compute prediction error


        #Discrimator

        pred = model(inputs)
        realpred = modelD(X.detach())
        fakepred = modelD(pred.detach())
        errD = lossD_fn(realpred, torch.ones_like(realpred)) + lossD_fn(fakepred, torch.zeros_like(fakepred))
        #errD = torch.mean(torch.relu(1.0 - realpred)) + torch.mean(torch.relu(1.0 + fakepred))
        #errD_fool = torch.mean(torch.relu(1.0 - fakepred))
        #errD_inv = lossD_fn(realpred, torch.zeros_like(realpred)) + lossD_fn(fakepred, torch.ones_like(fakepred))
        optimizerD.zero_grad()
        errD.backward()
        optimizerD.step()

        pred = model(inputs)
        fakepred = modelD(pred)
        errD_fool = lossD_fn(fakepred, torch.ones_like(fakepred))

        #label = torch.ones((batch_size,)) # fake labels are real for generator cost
        #output = modelD(pred)

        # MSE
        #loss = loss_fn(pred, X) + 0.00001*model.kl + errD_fool
        loss = loss_fn(pred, X) + 0.0001 * model.kl + 0.1*errD_fool
        #BCE
        #loss = loss_fn(pred, X) + model.kl + lossG_fn(output, label) #+ 0.001*loss_kl(pred, X)


        # Backpropagation


        #torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  Disloss:  {errD.item():>7f} [{current:>5d}/{size:>5d}]")


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
            test_loss += loss_fn(pred, X).item() + model.kl
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        #correct /= size
        print(f"testloss: {test_loss:>7f}")
        out_img = torch.squeeze(pred.cpu().data)
        print(out_img.size())

    for i in range(10):
        plt.subplot(1, 3, 1)
        plt.imshow(torch.squeeze(inputs[i]).cpu().numpy(), cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(torch.squeeze(X[i]).cpu().numpy(), cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(out_img[i].numpy(), cmap='gray')
        #print(inputs[i].size())
        state = model.get_latent(inputs[i]).cpu().detach().numpy()
        #print(str(model.get_latent(inputs[i]).cpu().detach().numpy()))
        #string = str(model.get_latent(inputs[i]).cpu().detach().numpy())
        #txt = 'Sate Vector : ' + string
        #fidelity = np.empty((1,5))
        for j in range(10):
            print( "Fidelity a "+ str(y[i]) + " vs  " + str(y[j])+ " " + str(np.square(np.abs(np.dot(state, model.get_latent(inputs[j]).cpu().detach().numpy().transpose())))))

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

epochs = 50
for t in range(epochs):
    print(f"Epoch Noise {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,noise=True)
    test(test_dataloader, model, loss_fn,noise=True)
print("Done!")
