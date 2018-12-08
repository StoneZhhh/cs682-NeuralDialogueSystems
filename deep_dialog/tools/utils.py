import torch.nn as nn
from torch import optim
from torch.nn import init

import numpy as np

'''
NLU utils
'''


def mergeDicts(d0, d1):
    """ for all k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
    for k in d1:
        if k in d0: d0[k] += d1[k]
        else: d0[k] = d1[k]


def initialize_weights(model):
    if isinstance(model, nn.Linear) or isinstance(model, nn.ConvTranspose2d):
        init.xavier_uniform_(model.weight.data)


def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    return optimizer


'''
Dialogue System utils
'''


def unique_states(training_data):
    unique = []
    for datum in training_data:
        if contains(unique, datum[0]):
            pass
        else:
            unique.append(datum[0].copy())
    return unique


def contains(unique, candidate_state):
    for state in unique:
        if np.array_equal(state, candidate_state):
            return True
        else:
            pass
    return False
