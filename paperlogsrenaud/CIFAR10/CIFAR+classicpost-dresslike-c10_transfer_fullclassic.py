from typing import Optional, Union
from pathlib import Path
import sys

import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
from torch.utils.tensorboard import SummaryWriter

from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput

sys.path.append(str(Path(__file__).resolve().parents[2]))
from model_paths import registered_model_path

import numpy as np
import math
torch.autograd.set_detect_anomaly(True)
import os
import datetime
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
path =  "paperlogs/CIFAR10/CC_classicpost-c10-tranfer-fullclassic/"+date+"/"
os.makedirs(path)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("device =", device, " device count ", torch.cuda.device_count())
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Download training data from open datasets.
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=Compose([Resize((128,128)),ToTensor()]), #ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=Compose([Resize((128,128)),ToTensor()]), #ToTensor(),
)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

print(len(training_data))
print(len(training_data.classes))
batch_size = 128
train_set, val_set = torch.utils.data.random_split(training_data, [40000, 10000])

# Create data loaders.
train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


"""
# replace by an import ...
class quantumautoencoder(AutoencoderKL):



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

        # Split the flattened tensor into two halves
        real_values, complex_values = torch.split(z_flattened, z_flattened.size(1) // 2, dim=1)

        # Create a complex tensor
        complex_tensor = torch.complex(real_values, complex_values)

        states = torch.nn.functional.normalize(complex_tensor, dim=1)

        if not self.Tomography:
            zz = torch.real(torch.square(torch.abs(states)))
            zz = torch.stack([zz, zz], dim=1)
        else:
            zz = torch.stack([states.real, states.imag], dim=1)

        zz = zz.reshape(z.shape)
        dec = self.decode(zz).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec), kl_div, states

    def get_latent(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = True,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ):
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

        # Split the flattened tensor into two halves
        real_values, complex_values = torch.split(z_flattened, z_flattened.size(1) // 2, dim=1)

        # Create a complex tensor
        complex_tensor = torch.complex(real_values, complex_values)

        states = torch.nn.functional.normalize(complex_tensor, dim=1)

        if not self.Tomography:
            zz = torch.real(torch.square(torch.abs(states)))
            zz = torch.stack([zz, zz], dim=1)
        else:
            zz = torch.stack([states.real, states.imag], dim=1)

        zz = zz.reshape(z.shape)
        return zz

    def set_Tomo(self, Tomography=False):
        self.Tomography = Tomography
        return self.Tomography

    def set_dev(self, dev, Tomography=False):
        self.dev = dev
        if Tomography:
            self.qlayer = qml.qnn.TorchLayer(self.contruct_circuit_state(), weight_shapes={})
        else:
            self.qlayer = qml.qnn.TorchLayer(self.contruct_circuit_prob(), weight_shapes={})


    def contruct_circuit_state(self):
        @qml.qnode(self.dev, diff_method="backprop", interface="torch", wires=self.wires)
        def circuit(inputs):
            qml.QubitStateVector(inputs, wires=self.wires)
            return qml.state()

        return circuit

    def contruct_circuit_prob(self):
        @qml.qnode(self.dev, diff_method="backprop", interface="torch", wires=self.wires)
        def circuit(inputs):
            qml.QubitStateVector(inputs, wires=self.wires)
            return qml.probs(wires=self.wires)

        return circuit

    def contruct_circuit_Hardamard_prob(self):
        @qml.qnode(self.dev, diff_method="backprop", interface="torch", wires=self.wires)
        def circuit(inputs):
            qml.QubitStateVector(inputs, wires=self.wires)
            for i in range(self.wires):
                qml.Hadamard(wires=i)
            return qml.probs(wires=self.wires)

        return circuit
"""

class CircuitCentric(nn.Module):
    def __init__(self, qubits=7, n_layer=20):
        super(CircuitCentric, self).__init__()
        self.dev = qml.device('default.qubit', wires=qubits)
        self.n_layers = n_layer
        self.n_qubits_ = qubits
        self.wires = np.arange(qubits)
        self.final = nn.Linear(2*(2**qubits), 10) # nombre de classes, todo as a variable


        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits_
        )

        #self.weights = torch.randn(size=shape, requires_grad=True)
        #b = torch.tensor(0.01)
        #self.register_parameter( name="circuit_weights" , param= torch.nn.Parameter(self.weights))
        #self.register_parameter( name='circuit_bias', param= torch.nn.Parameter(b))
        #@qml.qnode(self.dev, interface='torch', diff_method="backprop", wires=self.wires)
        #def circuit_classifier(inputs, weights):
            #qml.QubitStateVector(inputs, wires=self.wires)
            #qml.AmplitudeEmbedding(inputs, pad_with=0.0, wires=self.wires, normalize=True)

            #print("Weights used in StronglyEntanglingLayers:", weights)
            #qml.StronglyEntanglingLayers(weights, wires=self.wires)
            #return qml.expval(qml.PauliZ(0))
            #return qml.probs(self.wires)

        #print(qml.grad(circuit_classifier))

        self.qlayer = qml.qnn.TorchLayer(self.construct_circuit(), weight_shapes={"weights": shape}).to(device)
        #self.circuit = self.construct_circuit()
        #print(qml.draw(self.qlayer.qnode()))
        #print(self.qlayer)

    def forward(self, x):
        output = self.qlayer(x)
        c = torch.cat([torch.real(output), torch.imag(output)], dim=1)
        # print(c.shape) 
        output = self.final(c)
        return output




    def construct_circuit(self):
        @qml.qnode(self.dev, interface='torch', diff_method="backprop", wires=self.wires)
        def circuit_classifier(inputs,weights):
            #print(self.wires)
            qml.QubitStateVector(inputs, wires=self.wires)
            #qml.AmplitudeEmbedding(inputs,pad_with=0.0, wires=self.wires, normalize=True)

            #print("Weights used in StronglyEntanglingLayers:", weights)
            ### /// qml.StronglyEntanglingLayers(weights, wires=self.wires)
            #do i need to to tensor this ?
            #return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires]
            return qml.probs(self.wires)

        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.n_layers, n_wires=self.n_qubits_
        )

        #weights = torch.randn(size=shape)
        #inputs = torch.randn(size=(128,1024))
        #print(qml.draw(circuit_classifier)(inputs,weights))
        return circuit_classifier

def train(dataloader, model,model_vae,loss_fn, optimizer, epoch=0):
    size = len(dataloader.dataset)
    model.train()
    epoch += 1
    model_vae.eval()

    for batch, (X, y) in enumerate(dataloader):
        #print("y:", y, y.shape)
        X = X.view(-1,3,128,128).to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            state = model_vae.get_latent(X)

        ##print("shape", state.shape)
        #state = state.unsqueeze(-1)
        #copies = torch.einsum("bi,bj->bij",state,state).view(-1,state.shape[-1]*state.shape[-1])
        ## out = torch.transpose(model(state),1,0) # pas vraiment un etat cela dit.... (il a ete mesure)
        out = model(state)
        #print(out.grad_fn)

        ##print("out", out.shape, torch.sum(torch.isnan(out)))
        ##print("y", y.shape, torch.sum(torch.isnan(y)))
        
        #out = out.unsqueeze(-1)
        #torch.abs?
        loss = loss_fn(out, y.to(device)) ###.to(torch.float32)) ##  torch.nn.functional.one_hot(y.to(device)).to(torch.float32))

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
            state = model_vae.get_latent(X)

            #copies = torch.einsum("bi,bj->bij", state, state).view(-1, state.shape[-1] * state.shape[-1])
            pred = model(state) ### torch.abs(model(copies)) #the complex value is zero
            test_loss += loss_fn(pred, y.to(device)).item() ## .to(torch.float32)).item() ## .item()? torch.nn.functional.one_hot(y.to(device)).to(torch.float32)).item()
            correct += (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


if __name__ == '__main__':
    qubit = 11 # -> so 'state' is 2048
    #from variationalautoencodertestpennylane import NeuralNetwork
    from  vaequantumhugface_cifar10_transfer import quantumautoencoder
    modelVAE = quantumautoencoder(
        in_channels=3,
        out_channels=3,
        sample_size=32, # chamge later to 128
        block_out_channels=(32, 32, 64),
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
    ).to(device)
    modelVAE.set_Tomo(False)
    print(modelVAE) ##  NeuralNetwork(wires=qubit/2).to(device)
    print("device =", device, " device count ", torch.cuda.device_count())
    modelVAE.load_state_dict(
        torch.load(registered_model_path("cifar10_autoencoderkl"), map_location=device)
    )

    loss_fn = nn.CrossEntropyLoss() # nn.MSELoss() ##  nn.CrossEntropyLoss()

    model = CircuitCentric(qubit).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003) # 0.001
    epochs = 1000
    best_val_loss = float('inf')  # Initialize best validation loss as infinity
    best_val_Epoch = float('inf')  # Initialize best validation loss as infinity
    writer = SummaryWriter(path + 'loss_logging')  # TensorBoard writer

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, modelVAE, loss_fn, optimizer, t)
        val_loss, val_accuracy = test(val_dataloader, model, modelVAE, loss_fn)

        # Log validation loss to TensorBoard
        writer.add_scalar('Validation Loss', val_loss, t)
        writer.add_scalar('Validation Accuracy', val_accuracy, t)

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_accuracy = val_accuracy
            best_val_Epoch = t
            torch.save(model.state_dict(), path + "best_model.pth")
            print(
                f"New best validation loss: {best_val_loss:.4f}, accuracy: {best_val_loss_accuracy:.4f} - Model saved.")

        # After training, run the test function with test data

    model.load_state_dict(torch.load(path + "best_model.pth"))

    test_loss, test_accuracy = test(test_dataloader, model, modelVAE, loss_fn)
    writer.add_scalar('Test Loss', test_loss, best_val_Epoch)
    writer.add_scalar('Test Accuracy ', test_accuracy, best_val_Epoch)
    print(f"Test Loss after training: {test_loss:.4f} , accuracy: {test_accuracy:.4f}")
