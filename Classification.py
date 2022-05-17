import tensorboard

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from models import MLP, CNN, Resnet
from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
epochs = 50

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),

     ]
)
data_train = datasets.MNIST(root='data/',
                          train=True,
                          transform=transforms,
                          download=True)

data_test = datasets.MNIST(root='data/',
                         train=False,
                         transform=transforms,
                         download=True)

loader = DataLoader(dataset=data_train,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

network = Resnet().to(device)
optimizer = optim.Adam(network.parameters(), lr=lr)
loss = nn.CrossEntropyLoss().to(device)
# writer_real = SummaryWriter(f"logs/")

for epoch in range(epochs):
    total_cost = 0
    for _, (X, Y) in enumerate(loader):
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        pred = network(X)
        cost = loss(pred, Y)
        cost.backward()
        optimizer.step()

        total_cost += cost

    average_cost = total_cost/len(data_train)
    print(f'epoch: {epoch}, average cost: {average_cost}')




























