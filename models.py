'''
this code is to classify Mnist datasets
'''

import torch
import torch.nn as nn
from torch.nn import functional as F


# MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, input_img):
        x = input_img.view(input_img.size(0), -1)
        return self.model(x)


# CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (4, 4), (2, 2)),
            nn.ReLU()
        )
        self.linears = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, input_img):
        x = self.convs(input_img)
        x = x.view(x.size(0), -1)
        x = self.linears(x)
        return x



# Resnet
class Blocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(32)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, input_img):
        residual = input_img
        x = self.conv1(input_img)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = F.relu(x)
        return x


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Conv2d(1, 32, stride=(1,1), kernel_size=(3,3), padding='same')
        for block in range(2):
            setattr(self, "block{}".format(block), Blocks())
        self.output_conv = nn.Conv2d(32, 1, (3,3), (1,1), padding='same')
        self.fc = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, input_img):
        x = self.input_conv(input_img)
        for block in range(2):
            x = getattr(self, "block{}".format(block))(x)
        x = self.output_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x















