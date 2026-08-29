from typing import Optional, Union, Callable
import sys
from taming.modules.losses.lpips import LPIPS
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from functools import partial

import datetime
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
path = "paperlog/pretrained/vae kl  cifar 10 datareupload/"+date + "/"
#os.makedirs(path)
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
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


batch_size = 128

train_set, val_set = torch.utils.data.random_split(training_data, [40000, 10000])


# Create data loaders.
train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
class quantumautoencoder(AutoencoderKL):

    def __init__(self, n_quantum_layers=10, n_qubits=8, Tomography=False, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Forward all parent class parameters
        self.n_quantum_layers = n_quantum_layers
        self.n_qubits = n_qubits
        self.Tomography = Tomography
        self.wires = np.arange(n_qubits)
        self.dev = qml.device('default.qubit', wires=self.wires)
        self.shapeweight = (self.n_qubits, self.n_quantum_layers + 1, 3)
        self.shapeinput = (self.n_quantum_layers + 1, self.n_qubits * 3)

        # Projection layer: maps encoder output to quantum circuit input size
        self.project_to_quantum = None
        self.project_from_quantum = None
        self.qlayer = qml.qnn.TorchLayer(self.contruct_data_reuploding_circuit(), weight_shapes={"weights":  self.shapeweight }).to(device)


    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:

        x = sample
        posterior = self.encode(x).latent_dist
        kl_div = -0.5 * torch.sum(1 + posterior.logvar - posterior.mean.pow(2) - posterior.var)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        # Flatten z into batch, data
        old_shape = z.shape
        z_flattened = z.reshape(z.size(0), -1)

        # Dynamically initialize projection layer if input size is unknown
        if self.project_to_quantum is None:
            self.project_to_quantum = nn.Linear(
                in_features=z_flattened.size(1),
                out_features=self.n_qubits * 3 * (self.n_quantum_layers + 1),
            ).to(z.device)

        if self.project_from_quantum is None:
            self.project_from_quantum = nn.Linear(
                in_features=self.n_qubits,
                out_features=z_flattened.size(1),
            ).to(z.device)
        # Project z to quantum input size
        quantum_inputs = self.project_to_quantum(z_flattened)


        z = self.qlayer(quantum_inputs)


        zz = self.project_from_quantum(z)
        dec = self.decode(zz.reshape(old_shape)).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec), kl_div, quantum_inputs, z_flattened



    def contruct_data_reuploding_circuit(self):
        #@partial(qml.batch_input, argnum=1)
        @qml.qnode(self.dev, diff_method="backprop", interface="torch", wires=self.wires, batch_size=128)
        def circuit(inputs, weights):
            # code taken from better than classical paper have to switch to angel embedings for batches
            for layer in range(self.n_quantum_layers):
                x_idx = 0 # idx for inputs
                qml.AngleEmbedding(inputs[:, x_idx: x_idx + self.n_qubits], wires=self.wires, rotation='X')
                qml.AngleEmbedding(inputs[:, x_idx + self.n_qubits: x_idx + 2 * self.n_qubits], wires=self.wires, rotation='Y')
                qml.AngleEmbedding(inputs[:, x_idx + 2 * self.n_qubits: x_idx + 3 * self.n_qubits], wires=self.wires,
                                   rotation='Z')

                x_idx += 3 * self.n_qubits

                for i, wire in enumerate(self.wires):
                    angles = weights[i, layer, :]
                    qml.Rot(*angles, wires=wire)
                if layer % 2 == 0:
                    qml.broadcast(qml.CZ, self.wires, pattern="double")
                else:
                    qml.broadcast(qml.CZ, self.wires, pattern="double_odd")

            # final reupload without CZs

            x_idx = 0
            w = weights[:,self.n_quantum_layers]
            anglesX = torch.add(inputs[:, x_idx: x_idx + self.n_qubits],w[:,0])
            anglesY = torch.add(inputs[:, x_idx + self.n_qubits: x_idx + 2 * self.n_qubits],w[:,1])
            anglesZ= torch.add(inputs[:, x_idx + 2 * self.n_qubits: x_idx + 3 * self.n_qubits],w[:,2])

            qml.AngleEmbedding(anglesX, wires=self.wires, rotation='X')
            qml.AngleEmbedding(anglesY, wires=self.wires, rotation='Y')
            qml.AngleEmbedding(anglesZ, wires=self.wires, rotation='Z')


            return [qml.expval(qml.PauliZ(wires=[i])) for i in self.wires]


        return circuit



def getnoise(inputs):
    noise = torch.zeros(inputs.size(0), 3, 32, 32)
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
        output, kl , states, z = model(inputs)
        pred = output.sample
        loss = torch.mean(loss_fn(pred.contiguous(), X.contiguous())) + 0.000000001*kl
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)

        #if batch == 100:
            #nb.plot_grad_flow(path,model.named_parameters(),"VAE", epoch)

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
            out, kl , states, z = model(inputs)
            pred = out.sample
            test_loss +=  torch.mean(loss_fn(pred.contiguous(), X.contiguous())) + 0.00000001 * kl
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        #correct /= size
        print(f"testloss: {test_loss:>7f}")
        out_img = torch.squeeze(pred.cpu().data)
        print(out_img.size())

    fig, axs = plt.subplots(10, 3, figsize=(10, 10))  # Create a figure and a set of subplots
    for i in range(10):
        axs[i, 0].imshow(torch.squeeze(inputs[i].cpu().detach()).numpy().transpose(1, 2, 0),)
        axs[i, 1].axis('off')
        axs[i, 1].imshow(torch.squeeze(X[i].cpu().detach()).numpy().transpose(1, 2, 0))
        axs[i, 1].axis('off')
        axs[i, 2].imshow(out_img[i].cpu().detach().numpy().transpose(1, 2, 0))
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

    # Force file write fl

    sys.stdout.flush()
    sys.stderr.flush()

    # Optional: Force file system sync (Linux/Unix only)
    os.sync()
    return test_loss
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
if __name__ == '__main__':

    model = quantumautoencoder(
        in_channels=3,
        out_channels=3,
        sample_size=32,
        block_out_channels=(32, 32, 64),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    ).to(device)
    print(model)
    # loss_fn = nn.BCELoss(reduction='sum')
    loss_val = 1000000
    loss_fn = LPIPS().to(device).eval()
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 1000
    #Tomography = False
    #if Tomography:
        #path += "/Tomography/"+date+"/"
   # else:
       # path += "/Prob/" + date + "/"
    os.makedirs(path)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, noise = (t > 6), epoch=t)
        loss = test(val_dataloader, model, loss_fn, epoch=t)
        if loss < loss_val:
            loss_val = loss
            torch.save(model.state_dict(), path+"autoencoderkl pretrain cifar datareupload.pt")
    model.load_state_dict(torch.load(path+"autoencoderkl pretrain cifar datareupload.pt"))
    test(test_dataloader, model, loss_fn, noise=True, epoch=epochs+1)
    print("Done!")