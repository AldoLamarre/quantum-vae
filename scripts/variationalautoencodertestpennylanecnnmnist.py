from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import Any
from torchvision.transforms import functional
import torchvision.transforms as transforms
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

train_set, val_set = torch.utils.data.random_split(training_data, [50000, 10000])
#train_set, val_set = torch.utils.data.random_split(training_data, [10000, 40000])

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



class FilterLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bflw, fs -> bslw ', input, self.weight)
class NeuralNetwork(nn.Module):
    def __init__(self, wires=8 , embedings = 1):
        super().__init__()
        #self.flatten = nn.Flatten()
        #self.n_estimation_wires = 3
        #self.n_target_wires = 2
        self.wires = np.arange(wires)
        self.dev = qml.device("default.qubit", wires = self.wires)
        #self.n_estimation_wires = 5
        #self.target_wires = [0, 1]
        self.encoder = nn.Sequential(
            # 28-3+2*1 + 1 = 28 -> 32x32x32
            # 32-3+2*1 + 1 = 32 -> 32x32x32
            nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=3, padding=0, stride=1, dilation=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1, stride=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2), #4x4
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2),  # 2x2
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2),  # 1x1
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=2 ** len(self.wires), kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=2 ** len(self.wires)),
            nn.ReLU(),

            #nn.MaxPool2d(2),
            #nn.Flatten(),
            #nn.Linear(8192, 4096),
            #nn.BatchNorm1d(4096),
            #nn.ReLU(),
            #nn.Linear(4096, 2048),
            #nn.BatchNorm1d(2048),
            #nn.ReLU(), # tanh vs relu ?
        )
        self.wrealmu = torch.Tensor(1024,2 ** len(self.wires))
        self.wrealsigma = torch.Tensor(1024, 2 ** len(self.wires))
        self.wimgmu = torch.Tensor(1024, 2 ** len(self.wires))
        self.wimgsigma = torch.Tensor(1024, 2 ** len(self.wires))

        self.realmu = nn.Sequential(
            FilterLinear(2 ** len(self.wires), 2 ** len(self.wires)),

        )
        self.realsigma = nn.Sequential(
            FilterLinear(2 ** len(self.wires), 2 ** len(self.wires)),

        )
        self.imgmu = nn.Sequential(
            FilterLinear(2 ** len(self.wires), 2 ** len(self.wires)),

        )
        self.imgsigma = nn.Sequential(
            FilterLinear(2 ** len(self.wires), 2 ** len(self.wires)),

        )
        self.normale = torch.distributions.Normal(0, 1)

        self.lk=0
        self.klvar = 0, 0, 0, 0



        self.decoder = nn.Sequential(
            #nn.Linear(2**len(self.wires), 2048),
            #nn.BatchNorm1d(2048),
            #nn.ReLU(),
            #nn.Unflatten(1, (8, 8, 16)),
            nn.ConvTranspose2d(in_channels=2 ** len(self.wires), out_channels=512, kernel_size=3, padding=1, stride=1,),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=0, stride=1, dilation=2),
            nn.Sigmoid(),
        )


        #weight_shapes = {}
        #self.qlayer = qml.qnn.TorchLayer(self.circuitl, v)
        #self.qlayer = qml.qnn.TorchLayer(self.contruct_circuit(), weight_shapes={})

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var * epsilon
        return z

    def get_last_layer(self):
        return self.decoder[-2].weight

    def get_last_layer_encoder(self):
        return self.encoder[-5].weight
    def forward(self, x):
        self.encodedstate = self.get_latent(x)
        #print(encode)
        #weight_shapes =#{}

        #print(inputs)

        #quantumlatent = self.qlayer(self.encodedstate)
        #print(quantumlatent)
        probabilities = torch.square(torch.abs(self.encodedstate))
        decode = self.decoder(probabilities)

        #print(phase_estimated)
        #print(finalinputs.shape)

        return decode.view(-1, 1, 28, 28) , self.encodedstate
    def get_latent(self, x):
        #x = self.flatten(x)
        encodetemp = self.encoder(x)
        #print(encodetemp.shape)
        realmu = self.realmu(encodetemp)
        realsigma = self.realsigma(encodetemp)
        imgmu = self.imgmu(encodetemp)
        imgsigma = self.imgmu(encodetemp)
        real = self.reparameterization(realmu, torch.exp(0.5 * realsigma))
        img = self.reparameterization(imgmu, torch.exp(0.5 * imgsigma))
        state = torch.complex(real, img) #view(-1, 8, 8, 2 ** len(self.wires))

        encode = torch.nn.functional.normalize(state, dim=1)
        self.klvar = realmu, realsigma, imgmu, imgsigma
        self.logvar = torch.complex(realsigma,imgsigma)

        #self.kl = - 0.5 * (torch.sum(1 + realmu - realmu.pow(2) - realsigma.exp()) +
        #                   torch.sum(1 + imgmu - imgmu.pow(2) - imgsigma.exp()))
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
            #qml.DepolarizingChannel(p=0.1, wires=self.wires)
            #for i in self.wires:
                #qml.BitFlip(0.21, i)
            return qml.probs(wires=self.wires)
            #return qml.expval()
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
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()

            ############################
        )

    def forward(self, input):
        #####################################
        # Change the shape of output if necessary #

        #####################################
        input = input.view(-1, 32 * 32)
        output = self.main(input)

        #####################################
        # Change the shape of output if necessary #

        #####################################
        output = output.squeeze(dim=1)

        return output

nc = 1 # number of channels, RGB
nz = 100 # input noise dimension
ngf = 64 # number of generator filters
ndf = 64 #number of discriminator filters
class DiscriminatorCNN(nn.Module):
    def __init__(self):
        super(DiscriminatorCNN, self).__init__()
        self.main = nn.Sequential(
            ## input is (nc) x 64 x 64
            #nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32 // 28x28
            nn.Conv2d(nc, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Dropout(0.3),
            # state size. (ndf*8) x 4 x 4 ->  28x28 -> 3x3 kernel must be 3
            nn.Conv2d(ndf * 8, 1, 3, 1, 0, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)
nbwires=5
model = NeuralNetwork(wires=nbwires).to(device)
#modelDs = Discriminator().to(device)
#modelDCNN = DiscriminatorCNN().to(device)
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.relu(1. - logits_real))
    loss_fake = torch.mean(torch.nn.functional.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss
print(model)
#print(model.decoder[6])
#print(model.decoder[-2])
#lossD_fn = nn.BCELoss()
#lossG_fn =
#loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss(reduction="sum")
#loss_kl = nn.KLDivLoss(reduction="batchmean")
#optimizer = torch.optim.NAdam(model.parameters(), lr=0.00001)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
#optimizerD = torch.optim.Adam(model.parameters(), lr=0.0002)#must have low learning rate 0.0002
def getnoise(inputs):
    noise = torch.zeros(inputs.size(0), 3, 32, 32)
    nn.init.normal_(noise, 0, 0.1)
    noise = noise.to(device)
    noise_inputs = inputs + noise
    noise_inputs = torch.clamp(noise_inputs, 0, 1) # for rgb
    return noise_inputs

def train(dataloader, model, loss_fn, optimizer, noise=False, discriminatoronly = False, ganonly=False, dcnn=False):
    size = len(dataloader.dataset)
    model.train()

   # if dcnn:
        #modelD = modelDCNN
    #else:
        #modelD = modelDs
    #modelD.train()


    for batch, (X, y) in enumerate(dataloader):
        X =  X.to(device)
        if noise:
            inputs = getnoise(X)
        else:
            inputs = X
        # zero the parameter gradients
        optimizer.zero_grad()
        # Compute prediction error
        pred, quantumstate = model(inputs)
        last_layer = model.get_last_layer()
        #Discrimator
        #modelD.zero_grad()
        batch_size = inputs.size(0)
        label = torch.ones((batch_size,))# real label\
        #if dcnn:
        #    resizeX= transforms.functional.resize(X, size=28)
        #else:
        #    resizeX = X
        #resizeX = X
        #realoutput = modelD(resizeX)
        #noisy = getnoise(X) - X
        #errD_real = vanilla_d_loss(realoutput, label)

        #label = torch.zeros((batch_size,))
        #if dcnn:
        #    resizepred= transforms.functional.resize(pred, size=28)
        #else:
        #    resizepred = pred
        #resizepred = pred
        #outputvae = modelD(noisy.detach())
        #errD_fake = lossD_fn(outputvae, label)
        #meanfake = torch.mean(outputvae)
       #meanreal = torch.mean(realoutput)


        #errD = vanilla_d_loss(realoutput,outputvae)
        #errD = errD_real + errD_fake
        #errD.backward()
        #optimizerD.step()


        if discriminatoronly == False:
            #label = torch.ones((batch_size,)) # fake labels are real for generator cost
            #output = modelD(pred)

            # MSE
            #loss = loss_fn(pred, X) + 0.00001*model.kl + 0.01 * lossG_fn(output, label) #+ 0.001*loss_kl(pred, X)
            #BCE
            #print(X.shape)
            #print(pred.shape)
            #recloss = torch.abs(X - pred)
            #nllloss = loss_fn(pred, X)
            #logmean = torch.mean(model.logvar)

            #nllloss = torch.mean(torch.abs(recloss / torch.exp(logmean) + logmean))
            #nllloss = torch.mean(recloss)
            #nllloss = nllloss + 0.000001*loss_fn(pred, X)
            nllloss = loss_fn(pred, X)
            #gloss = -torch.mean(output)
            #nll_grad = torch.autograd.grad(nllloss, last_layer, retain_graph=True)[0]
            #g_grad = torch.autograd.grad(gloss, last_layer, retain_graph=True)[0]
            #d_weight = torch.norm(nll_grad) / ( torch.norm(g_grad) + 1e-4)
            #d_weight = torch.clamp(d_weight, 0.0, 1e-4)

            # Minimise cycle trace distance sqrt(1- fidelity) to benefits from quantum latent spaces
            # Makes the autoencoder much worse
            statecycle = model.get_latent(pred)
            #print(quantumstate.shape)
            #print(statecycle.shape)
            #layer_cycle = model.get_last_layer_encoder()
            #fidelity = torch.square(torch.abs(torch.matmul(quantumstate,torch.t(statecycle))))
            statecycle = torch.squeeze(statecycle)
            quantumstate = torch.squeeze(quantumstate)
            fidelity = qml.math.fidelity(qml.math.dm_from_state_vector(quantumstate),qml.math.dm_from_state_vector(statecycle))
            #cycleloss = torch.mean(torch.sqrt(1.0-fidelity))
            #cycleloss = -torch.mean(fidelity)
            cycleloss = torch.mean(1.0-fidelity)
            #nll_cyclegrad = torch.autograd.grad(nllloss, layer_cycle, retain_graph=True)[0]
            #cyclegrad = torch.autograd.grad(cycleloss, layer_cycle, retain_graph=True)[0]


            #c_weight = torch.norm(nll_cyclegrad) / (torch.norm(cyclegrad) + 1e-4)
            #c_weight = torch.clamp(c_weight, 0.0, 1e-4)

            #if noise:
                #loss = nllloss + 0.00001*model.kl + d_weight * gloss #+ 0.00001* c_weight* cycleloss
                #print("yes")
            #elif ganonly==True:
                #loss = gloss

            loss = nllloss + 0.000001*model.kl + 0.00000001 * cycleloss
            #loss = nllloss + model.kl
            # Backpropagation
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"nllloss: {nllloss:>7f}  kl: {model.kl:>7f}  weighkl: {0.000001*model.kl:>7f} ")
            #print(f"loss: {loss:>7f}  Disloss:  {errD.item():>7f} MeanReal:  {meanreal.item():>7f} MeanFake:  {meanfake.item():>7f} [{current:>5d}/{size:>5d}]")
        #else:
            #if batch % 100 == 0:
                #loss, current = errD.item(), (batch + 1) * len(X)
                #print(f"Disloss:  {loss:>7f} MeanReal:  {meanreal.item():>7f} MeanFake:  {meanfake.item():>7f} [{current:>5d}/{size:>5d}]")


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
            pred,latent = model(inputs)
            test_loss += loss_fn(pred, X).item()  #+ 0.00001 * model.kl
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        #correct /= size
        print(f"Valloss: {test_loss:>7f}")
        out_img = torch.squeeze(pred.cpu().data)
        #out_latent = torch.squeeze(latent.cpu().data)
        print(out_img.size())

    for i in range(10):
        plt.subplot(1, 3, 1)
        plt.imshow(inputs[i].cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 3, 2)
        plt.imshow(X[i].cpu().numpy().transpose(1, 2, 0))
        plt.subplot(1, 3, 3)
        #print(out_img[i].shape)
        plt.imshow(out_img[i].cpu().numpy().transpose(1, 2, 0))
        #reshapeinput = inputs[i].view(1,1,28,28)
        #state = model.get_latent(reshapeinput).cpu().detach().numpy()
        #print(str(model.get_latent(inputs[i]).cpu().detach().numpy()))
        #string = str(model.get_latent(inputs[i]).cpu().detach().numpy())
        #txt = 'Sate Vector : ' + string
        #fidelity = np.empty((1,5))
        #for j in range(10):
            #print( "Fidelity a "+ str(y[i]) + " vs  " + str(y[j])+ " " + str(np.square(np.abs(np.dot(out_latent[i].numpy(), out_latent[j].numpy().transpose())))))

        #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        #plt.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        plt.show()
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, ganonly=True)
    test(val_dataloader, model, loss_fn)
print("Done!")

epochs = 20
for t in range(epochs):
    print(f"Epoch Noise {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,noise=True)
    test(val_dataloader, model, loss_fn,noise=True)
print("Done!")
