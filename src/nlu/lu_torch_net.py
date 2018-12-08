from tools.utils import *

import gensim
import shutil

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.nn import MSELoss
import torch.nn.functional as F

import numpy as np

from models.wordvector.word_tokenizer import Tokenizer


# x is the input word sequence and y contains the associated slots
# use word vectors
# 10 layers lstm network
from nlu.base_wrapper import BaseWrapper

dtype = torch.FloatTensor


class LSTMWrapper(BaseWrapper):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__(input_size, hidden_size, output_size)

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh()
        )

        self.model.apply(initialize_weights)
        self.optimizer = get_optimizer(self.model.type(dtype))

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

    def save_checkpoint(self, path):
        torch.save({'model_state_dict': self.model.state_dict(), 'opt_state_dict': self.optimizer.state_dict()}, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['opt_state_dict'])
        self.model.eval()

    """ Activation Function: Sigmoid, or tanh, or ReLu """
    ''' Run on one sample in a batch '''
    def one_batch_train(self, Xs, Y):
        Ws = Xs['word_vectors']

        self.optimizer.zero_grad()
        outputs = self.model(Ws)
        mse_loss = MSELoss(outputs, Y)

        mse_loss.backward()
        self.optimizer.step()
        self.save_checkpoint('../../../data/nn/lstm.pth.tar')
        return self.model.data.cpu().numpy(), mse_loss.item()

    def one_batch_predict(self, Xs):
        Ws = Xs['word_vectors']
        return self.model(Ws)



