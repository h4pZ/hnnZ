"""Implementation of a variational autoencoder in pytorch 
https://arxiv.org/pdf/1312.6114.pdf"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms