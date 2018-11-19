"""Implementation of a variational autoencoder in pytorch
https://arxiv.org/pdf/1312.6114.pdf"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


print("Downloading the dataset")
# Download and load the training data
download_path = '/home/h4pz/Downloads/F_MNIST_data/'
trainset = datasets.FashionMNIST(
    download_path,
    download=True,
    train=True)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST(
    download_path,
    download=True,
    train=False)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=True)

print("ASD")
image, label = next(iter(trainloader))
print(type(image))
print(type(label))
