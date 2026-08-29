from typing import Optional, Union

from taming.modules.losses.lpips import LPIPS
import torch
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DecoderOutput
from torch import nn
from torch.nn.modules.module import T
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Resize
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
path = "paperlogs/hugging face qvae/qvae kl cifar10_transfer/"  #+date+"/"
#os.makedirs(path)
torch.autograd.set_detect_anomaly(True)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("device = ", device)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Download training data from open datasets.
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=Compose([Resize((256,256)),ToTensor()]), ## ToTensor(),
)


# Download test data from open datasets.
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=Compose([Resize((256,256)),ToTensor()]), ## ToTensor(),
)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

batch_size = 16 #128 # 2048 # 1024 # 1024 # 1024 # 1024 # 1024 # 1024 # 1024 # 1024 # 1024 # 1024 # 256 # 128

print(len(training_data))
print(len(training_data.classes))
train_set, val_set = torch.utils.data.random_split(training_data, [40000, 10000])


# Create data loaders.
train_dataloader = DataLoader(train_set, batch_size=batch_size)
val_dataloader = DataLoader(val_set, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
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

        return states
        #if not self.Tomography:
        #    zz = torch.real(torch.square(torch.abs(states)))
        #    zz = torch.stack([zz, zz], dim=1)
        #else:
        #    zz = torch.stack([states.real, states.imag], dim=1)
        #
        #zz = zz.reshape(z.shape)
        #return zz

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

def getnoise(inputs):
    noise = torch.zeros(inputs.size(0), 3, 256, 256)
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
        output, kl , states = model(inputs)
        pred = output.sample
        loss = torch.mean(loss_fn(pred.contiguous(), X.contiguous())) + 0.00000001*kl
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
            out, kl , states = model(inputs)
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
    return test_loss
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
if __name__ == '__main__':

    model = quantumautoencoder(
        in_channels=3,
        out_channels=3,
        sample_size=32, # et pas 32, taille des images
        block_out_channels=(32, 32, 64), # consider making it bigger and profonder: (32, 32, 32, 64) + les blocks (attention etc.)
        down_block_types=("DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"),
    ).to(device)
    print(model)
    # loss_fn = nn.BCELoss(reduction='sum')
    loss_val = 1000000
    loss_fn = LPIPS().to(device).eval()
    # loss_kl = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    epochs = 10000
    Tomography = False
    if Tomography:
        path += "/Tomography/"+date+"/"
    else:
        path += "/Prob/" + date + "/"
    os.makedirs(path)
    model.set_Tomo(Tomography)

    path2 = 'results/'
    model.load_state_dict(torch.load(path2+"autoencoderklimagenet3_ref.pt", map_location=torch.device('cuda')))
    print("model loaded")
    #model.load_state_dict(torch.load("results/autoencoderklcifar100.pt"))
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, noise = (t > 100), epoch=t)
        loss = test(val_dataloader, model, loss_fn, epoch=t)
        if loss < loss_val:
            loss_val = loss
            torch.save(model.state_dict(), path+"autoencoderklcifar10_transfer.pt")
    model.load_state_dict(torch.load(path+"autoencoderklcifar10_transfer.pt"))
    test(test_dataloader, model, loss_fn, noise=True, epoch=epochs+1)
    print("Done!")
