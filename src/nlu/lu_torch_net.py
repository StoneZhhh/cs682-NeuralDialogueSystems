import torch
import torch.nn as nn

import numpy as np


# x is the input word sequence and y contains the associated slots
# use word vectors
# 10 layers lstm network
def lstm_net():
    model = nn.Sequential(
        nn.LSTM(100, 10, num_layers=10),
    )

    return model

