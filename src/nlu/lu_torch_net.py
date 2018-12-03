import shutil

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim

import numpy as np


# x is the input word sequence and y contains the associated slots
# use word vectors
# 10 layers lstm network
def lstm_net():
    model = nn.Sequential(
        nn.LSTM(100, 10, num_layers=10),
    )

    return model


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform_(m.weight.data)


def get_adam_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer


def save_checkpoint(model, optimizer, path):
    torch.save({'model_state_dict': model.state_dict(), 'opt_state_dict': optimizer.state_dict()}, path)


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    model.eval()
    return model, optimizer

