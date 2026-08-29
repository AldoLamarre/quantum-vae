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
import networkblocks as nb
torch.autograd.set_detect_anomaly(True)
import os
import datetime
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
path = ("log/PCAdecoder/MNIST 5 QUBIT/")+date+"/"
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
train_set, val_set = torch.utils.data.random_split(training_data, [50000, 10000])

# Create data loaders.
train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=batch_size)
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
class PCA_MNIST:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.eigenvectors = None

    def fit(self, training_data):
        # Flatten the images
        data = training_data.data.view(len(training_data), -1) / 255.0

        # Compute the mean and subtract it from the data
        self.mean = torch.mean(data, dim=0)
        data -= self.mean

        # Compute the covariance matrix
        cov_matrix = torch.mm(data.t(), data) / data.shape[0]

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = torch.linalg.eig(cov_matrix)

        # Sort the eigenvectors by decreasing eigenvalues
        _, indices = torch.sort(torch.real(eigenvalues), descending=True)
        self.eigenvectors = torch.real(eigenvectors[:, indices[:self.n_components]])

    def transform(self, images):
        # Flatten the images and subtract the mean
        images = images.view(images.shape[0], -1) / 255.0 - self.mean.to(images.device)

        # Add an extra dimension to the image tensor
        #images = torch.unsqueeze(images, 1)

        # Project the images onto the principal components
        return torch.mm(images, self.eigenvectors.to(images.device))

    def inverse_transform(self, transformed_image):
        # Reconstruct the original image is it okay to take abs ?
        return torch.mm(torch.tensor(torch.real(torch.abs(transformed_image))), self.eigenvectors.t().to(transformed_image.device))

class Pcadecoder(nn.Module):
    def __init__(self, pca, wires=8):
        super().__init__()
        self.flatten = nn.Flatten()
        #self.n_estimation_wires = 3
        #self.n_target_wires = 2
        self.wires = np.arange(wires)
        self.dev = qml.device("default.qubit", wires = self.wires)
        #self.n_estimation_wires = 5
        #self.target_wires = [0, 1]
        self.pca = pca



        self.decoder = nn.Sequential(
            nn.Linear(2**len(self.wires), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),
        )
        #weight_shapes = {}
        #self.qlayer = qml.qnn.TorchLayer(self.circuitl, v)
        self.qlayer = qml.qnn.TorchLayer(self.contruct_circuit(), weight_shapes={})


    def forward(self, x):
        data = self.pca.transform(x)

        normalized_data = torch.nn.functional.normalize(data)
        normalized_data = torch.complex(normalized_data, torch.zeros_like(normalized_data))
        #print(encode)
        #weight_shapes =#{}

        #print(inputs)

        quantumlatent = self.qlayer(normalized_data)
        #print(quantumlatent)
        decode = self.decoder(torch.abs(quantumlatent))
        #print(phase_estimated)
        #print(finalinputs.shape)

        return decode.view(-1, 1, 28, 28),quantumlatent
    def get_latent(self, x):
        data = self.pca.transform(x)

        normalized_data = data / torch.linalg.norm(data)
        normalized_data = torch.complex(normalized_data, torch.zeros_like(normalized_data))
        # print(encode)
        # weight_shapes =#{}

        # print(inputs)

        #quantumlatent = self.qlayer(encode)
        return normalized_data

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




def getnoise(inputs):
    noise = torch.zeros(inputs.size(0), 1, 28, 28)
    nn.init.normal_(noise, 0, 0.1)
    noise = noise.to(device)
    noise_inputs = inputs + noise
    return noise_inputs

def train(dataloader, model, loss_fn, optimizer, noise=False, epoch=0):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0.0
    for batch, (X, y) in enumerate(dataloader):
        X =  X.to(device)
        if noise:
            inputs = getnoise(X)
        else:
            inputs = X
        # zero the parameter gradients
        optimizer.zero_grad()
        # Compute prediction error
        pred,quantumstate = model(inputs)
        statecycle = model.get_latent(pred)

        #fidelity = torch.square(torch.abs(torch.matmul(quantumstate, torch.t(statecycle))))
        # cycleloss = torch.mean(torch.sqrt(1.0-fidelity))
        # cycleloss = -torch.mean(fidelity)
        #cycleloss = torch.mean(1.0 - fidelity)
        loss = loss_fn(pred, X)
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)

        if batch == 100:
            nb.plot_grad_flow(path,model.named_parameters(),"VAE", epoch)

        #for name, param in model.named_parameters():
            #if param.requires_grad:
                #if torch.any(param.grad > 0):
                    #print(name, param.grad)
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Average loss for the batch: {total_loss/size}")
def test(dataloader, model, loss_fn,noise=False,epoch=0):
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
            pred, state = model(inputs)
            test_loss += loss_fn(pred, X).item() #+ model.kl
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        #correct /= size
        print(f"testloss: {test_loss:>7f}")
        out_img = torch.squeeze(pred.cpu().data)
        print(out_img.size())

    fig, axs = plt.subplots(10, 3, figsize=(10, 10))  # Create a figure and a set of subplots
    for i in range(10):
        axs[i, 0].imshow(torch.squeeze(inputs[i].cpu().detach()).numpy(), cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 1].imshow(torch.squeeze(X[i].cpu().detach()).numpy(), cmap='gray')
        axs[i, 1].axis('off')
        axs[i, 2].imshow(out_img[i].cpu().detach().numpy(), cmap='gray')
        axs[i, 2].axis('off')
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
    plt.savefig(path + "Epoch " + str(epoch) + ".png")
    # plt.show()
    plt.close(fig)
    return test_loss
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
if __name__ == '__main__':
    qubits = 5
    # Create a PCA_MNIST object with 50 components
    pca_mnist = PCA_MNIST(2 ** qubits)

    # Fit the PCA model to the MNIST training data
    pca_mnist.fit(train_set.dataset)
    model = Pcadecoder(pca=pca_mnist, wires=qubits).to(device)
    print(model)
    # loss_fn = nn.BCELoss(reduction='sum')
    loss_val = 1000000
    loss_fn = nn.MSELoss()
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 1000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, noise = (t > 6), epoch=t)
        loss = test(val_dataloader, model, loss_fn, epoch=t)
        if loss < loss_val:
            loss_val = loss
            torch.save(model.state_dict(), "pcaautoencoder  five.pt")
    model.load_state_dict(torch.load("pcaautoencoder five.pt"))
    test(test_dataloader, model, loss_fn, noise=True, epoch=epochs+1)
    print("Done!")


