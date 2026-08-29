from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from src.quantum_vae.utils.phase_shadow_family import build_phase_shadow_data_bundle
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

batch_size = 32
mnist_bundle = build_phase_shadow_data_bundle(training_data, test_data, batch_size=batch_size)
train_set = mnist_bundle["train_set"]
val_set = mnist_bundle["val_set"]
train_dataloader = mnist_bundle["train_dataloader"]
val_dataloader = mnist_bundle["val_dataloader"]
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

#https://arxiv.org/pdf/math-ph/0609050.pdf orthogonal matrix generation
class NeuralNetwork(nn.Module):
    def __init__(self, nbwires=7, n_estimation_wires=4):
        super().__init__()
        self.flatten = nn.Flatten()
        self.n_estimation_wires = n_estimation_wires
        self.n_target_wires = nbwires - self.n_estimation_wires
        self.wires = np.arange(nbwires)
        self.dev = qml.device("default.qubit", wires = self.wires)
        #self.n_estimation_wires = 5
        self.target_wires = np.arange(self.n_target_wires)
        #print(self.target_wires)
        self.estimation_wires = self.n_target_wires+np.arange(n_estimation_wires)
        #print(self.estimation_wires)
        self.eigenvector_param_size = 2 ** self.n_target_wires
        self.unitary_param_size = 4 ** self.n_target_wires
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

        )
        self.vectorrealmu = nn.Sequential(
            nn.Linear(256, self.eigenvector_param_size),

        )
        self.vectorrealsigma = nn.Sequential(
            nn.Linear(256,  self.eigenvector_param_size),

        )
        self.vectorimgmu = nn.Sequential(
            nn.Linear(256,  self.eigenvector_param_size),

        )
        self.vectorimgsigma = nn.Sequential(
            nn.Linear(256, self.eigenvector_param_size),

        )

        self.unitaryrealmu = nn.Sequential(
            nn.Linear(256, self.unitary_param_size),

        )
        self.unitaryrealsigma = nn.Sequential(
            nn.Linear(256, self.unitary_param_size),

        )
        self.unitaryimgmu = nn.Sequential(
             nn.Linear(256, self.unitary_param_size),

        )
        self.unitaryimgsigma = nn.Sequential(
            nn.Linear(256, self.unitary_param_size),

        )

        self.unitarytaurealmu = nn.Sequential(
            nn.Linear(256,  self.eigenvector_param_size),

        )
        self.unitarytaurealsigma = nn.Sequential(
            nn.Linear(256,  self.eigenvector_param_size),

        )
        self.unitarytauimgmu = nn.Sequential(
            nn.Linear(256,  self.eigenvector_param_size),

        )
        self.unitarytauimgsigma = nn.Sequential(
            nn.Linear(256,  self.eigenvector_param_size),

        )
        self.normale = torch.distributions.Normal(0, 1)

        self.lk=0
        self.klvar = 0, 0, 0, 0



        self.decoder = nn.Sequential(
            nn.Linear(2**len(self.estimation_wires), 256),
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

    def get_last_layer(self):
        return self.decoder[-2].weight

    def get_last_layer_encoder(self):
        return self.encoder[-3].weight
    def forward(self, x):
        self.encodedtemp = self.get_encode(x)
        self.latenteigenvector = self.get_latent_vector(self.encodedtemp)
        self.latentunitary = self.get_latent_unitary(self.encodedtemp)
        print(self.latenteigenvector.shape)
        print(self.latentunitary.reshape(-1,4**self.n_target_wires).shape)
        inputs = torch.cat((self.latenteigenvector,self.latentunitary.reshape(-1,4**self.n_target_wires)),dim=1)
        #inputs = self.latenteigenvector, self.latentunitary
        #print(encode)
        #weight_shapes =#{}

        #print(inputs)

        quantumlatent = self.qlayer(inputs) # may have to cpu it.
        #print(quantumlatent)
        decode = self.decoder(torch.abs(quantumlatent))
        #print(phase_estimated)
        #print(finalinputs.shape)
        #no cycle loss
        return decode.view(-1, 1, 28, 28) #, self.encodedstate

    def get_encode(self, x):
        x = self.flatten(x)
        encodetemp = self.encoder(x)
        return encodetemp
    def get_latent_vector(self, x):
        realmu = self.vectorrealmu(x)
        realsigma = self.vectorrealsigma(x)
        imgmu = self.vectorimgmu(x)
        imgsigma = self.vectorimgmu(x)
        real = self.reparameterization(realmu, torch.exp(0.5 * realsigma))
        img = self.reparameterization(imgmu, torch.exp(0.5 * imgsigma))
        state = torch.complex(real, img)

        encode = torch.nn.functional.normalize(state, dim=1)
        self.klvar = realmu, realsigma, imgmu, imgsigma
        self.logvar = torch.complex(realsigma,imgsigma)

        self.klvector = - 0.5 * (torch.sum(1 + realsigma - realmu.pow(2) - realsigma.exp()) +
                           torch.sum(1 + imgsigma - imgmu.pow(2) - imgsigma.exp()))
        # print(encode)
        # weight_shapes =#{}

        # print(inputs)

        #quantumlatent = self.qlayer(encode)
        return encode


    # no complex cause of pauli parameters
    def get_latent_unitary(self, x):
        realmu = self.unitaryrealmu(x)
        realsigma = self.unitaryrealsigma(x)
        imgmu = self.unitaryimgmu(x)
        imgsigma = self.unitaryimgmu(x)
        real = self.reparameterization(realmu, torch.exp(0.5 * realsigma))
        img = self.reparameterization(imgmu, torch.exp(0.5 * imgsigma))
        taurealmu = self.unitarytaurealmu(x)
        taurealsigma = self.unitarytaurealsigma(x)
        tauimgmu = self.unitarytauimgmu(x)
        tauimgsigma = self.unitarytauimgmu(x)
        taureal = self.reparameterization(taurealmu, torch.exp(0.5 * taurealsigma))
        tauimg = self.reparameterization(tauimgmu, torch.exp(0.5 * tauimgsigma))
        matrix = torch.complex(real, img).view(-1,2**self.n_target_wires,2**self.n_target_wires)
        tau = torch.complex(taureal, tauimg)




        unitary = torch.linalg.householder_product(matrix, tau)
        # could use caley
        #unitary = torch.nn.utils.parametrizations.orthogonal(matrix, orthogonal_map="matrix_exp")

        #encode = torch.nn.functional.normalize(state, dim=1)
        encode = unitary
        #self.klvar = realmu, realsigma, imgmu, imgsigma
        #self.logvar = torch.complex(realsigma,imgsigma)
        self.klunitary = - 0.5 * (torch.sum(1 + realsigma - realmu.pow(2) - realsigma.exp()))

        #self.kl = - 0.5 * (torch.sum(1 + realmu - realmu.pow(2) - realsigma.exp()) +
                          # torch.sum(1 + imgmu - imgmu.pow(2) - imgsigma.exp()))
        # print(encode)
        # weight_shapes =#{}

        # print(inputs)

        #quantumlatent = self.qlayer(encode)
        return encode

    def contruct_circuit(self):
        # @qml.batch_input(argnum=1)
        #@qml.batch_input()
        @qml.qnode(self.dev, diff_method="backprop", interface="torch")
        def circuit(inputs):
            #qml.QubitStateVector(inputs[0:self.eigenvector_param_size], wires=self.target_wires) disable return
            print(inputs.shape)
            vector = inputs[:,0:self.eigenvector_param_size]
            matrix = inputs[:,self.eigenvector_param_size:self.unitary_param_size+self.eigenvector_param_size]
            print(vector.shape)
            print(matrix.shape)
            qml.QubitStateVector(vector, wires=self.target_wires)

            unitary = qml.QubitUnitary(U=matrix.view(-1,2**self.n_target_wires,2**self.n_target_wires), wires=self.target_wires)
            #oplist = self.compute_decomposition(self.wires, unitary, self.target_wires, self.estimation_wires)
            #qml.apply(oplist)
            qml.apply(unitary)
            # qml.QubitStateVector(torch.sqrt(eigenvector), wires=self.target_wires)
            # QuantumPhaseEstimation(
            # unitary,
            # estimation_wires=estimation_wires,
            # )
            return qml.probs(self.estimation_wires)

        return circuit

        # Stolen from pennylane source.

    @staticmethod
    def compute_decomposition(
            wires, unitary, target_wires, estimation_wires
    ):  # pylint: disable=arguments-differ,unused-argument
        r"""Representation of the QPE circuit as a product of other operators.

        .. math:: O = O_1 O_2 \dots O_n.


        .. seealso:: :meth:`~.QuantumPhaseEstimation.decomposition`.

        Args:
            wires (Any or Iterable[Any]): wires that the QPE circuit acts on
            unitary (Operator): the phase estimation unitary, specified as an operator
            target_wires (Any or Iterable[Any]): the target wires to apply the unitary
            estimation_wires (Any or Iterable[Any]): the wires to be used for phase estimation

        Returns:
            list[.Operator]: decomposition of the operator
        """

        op_list = [qml.Hadamard(w) for w in estimation_wires]
        pow_ops = (qml.pow(unitary, 2 ** i) for i in range(len(estimation_wires) - 1, -1, -1))
        op_list.extend(qml.ctrl(op, w) for op, w in zip(pow_ops, estimation_wires))
        op_list.append(qml.adjoint(qml.templates.QFT(wires=estimation_wires)))

        return op_list

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

model = NeuralNetwork(nbwires=8).to(device)
#modelD = Discriminator().to(device)
#qml.disable_return() # for cuda
def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(torch.nn.functional.relu(1. - logits_real))
    loss_fake = torch.mean(torch.nn.functional.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss
print(model)
#print(model.decoder[6])
#print(model.decoder[-2])
#lossD_fn = nn.BCELoss()
#lossG_fn =
loss_fn = nn.MSELoss()
#loss_fn = nn.BCELoss(reduction="sum")
#loss_kl = nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)
optimizerD = torch.optim.NAdam(model.parameters(), lr=1e-3)
def getnoise(inputs):
    noise = torch.zeros(inputs.size(0), 1, 28, 28)
    nn.init.normal_(noise, 0, 0.1)
    noise = noise.to(device)
    noise_inputs = inputs + noise
    return noise_inputs

def train(dataloader, model, loss_fn, optimizer, noise=False):
    size = len(dataloader.dataset)
    model.train()
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
        #pred, quantumstate = model(inputs)
        pred = model(inputs)
        last_layer = model.get_last_layer()
        #Discrimator
        #modelD.zero_grad()
        #batch_size = inputs.size(0)
        #label = torch.ones((batch_size,))# real label
        #realoutput = modelD(X)
        #errD_real = lossD_fn(realoutput, label)

        #label = torch.zeros((batch_size,))
        #outputvae = modelD(pred.detach())
        #errD_fake = lossD_fn(outputvae, label)
        #meanfake = torch.mean(outputvae)
        #meanreal = torch.mean(realoutput)


        #errD = hinge_d_loss(realoutput,outputvae)
        #errD.backward()
        #optimizerD.step()

        #label = torch.ones((batch_size,)) # fake labels are real for generator cost
        #output = modelD(pred)

        # MSE
        #loss = loss_fn(pred, X) + 0.00001*model.kl + 0.01 * lossG_fn(output, label) #+ 0.001*loss_kl(pred, X)
        #BCE
        #recloss = torch.abs(X - pred)
        #nllloss = loss_fn(pred, X)
        #logmean = torch.mean(model.logvar)

        #nllloss = torch.mean(torch.abs(recloss / torch.exp(logmean) + logmean))
        #nllloss = torch.mean(recloss)
        nllloss = loss_fn(pred, X)
        #gloss = -torch.mean(output)
        #nll_grad = torch.autograd.grad(nllloss, last_layer, retain_graph=True)[0]
        #g_grad = torch.autograd.grad(gloss, last_layer, retain_graph=True)[0]
        #d_weight = torch.norm(nll_grad) / ( torch.norm(g_grad) + 1e-4)
        #d_weight = torch.clamp(d_weight, 0.0, 1e-4)

        # Minimise cycle trace distance sqrt(1- fidelity) to benefits from quantum latent spaces
        # Makes the autoencoder much worse
        #statecycle = model.get_latent(pred)
        #print(quantumstate.shape)
        #print(statecycle.shape)
        #layer_cycle = model.get_last_layer_encoder()
        #fidelity = torch.square(torch.abs(torch.matmul(quantumstate,torch.t(statecycle))))
        #cycleloss = torch.mean(torch.sqrt(1.0-fidelity))
        #cycleloss = -torch.mean(fidelity)
        #cycleloss = torch.mean(1.0-fidelity)
        #nll_cyclegrad = torch.autograd.grad(nllloss, layer_cycle, retain_graph=True)[0]
        #cyclegrad = torch.autograd.grad(cycleloss, layer_cycle, retain_graph=True)[0]


        #c_weight = torch.norm(nll_cyclegrad) / (torch.norm(cyclegrad) + 1e-4)
        #c_weight = torch.clamp(c_weight, 0.0, 1e-4)

        loss = nllloss #+ 0.00001*(model.klvector + model.klunitary) #+ 0.00001 *c_weight* cycleloss

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
            test_loss += loss_fn(pred, X).item() #+ 0.00001 * (model.klvector + model.klunitary)
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
        #state = model.get_latent(inputs[i]).cpu().detach().numpy()
        #print(str(model.get_latent(inputs[i]).cpu().detach().numpy()))
        #string = str(model.get_latent(inputs[i]).cpu().detach().numpy())
        #txt = 'Sate Vector : ' + string
        #fidelity = np.empty((1,5))
        #for j in range(10):
            #print( "Fidelity a "+ str(y[i]) + " vs  " + str(y[j])+ " " + str(np.square(np.abs(np.dot(state, model.get_latent(inputs[j]).cpu().detach().numpy().transpose())))))

        #plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
        #plt.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')
        plt.show()
        #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

epochs = 5
for t in range(epochs):
    print(f"Epoch Noise {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer,noise=True)
    test(test_dataloader, model, loss_fn,noise=True)
print("Done!")
