import nlu.lu_torch_net as lunet
from models.wordvector.tokenizer import Tokenizer

import torch
import torch.nn as nn
from torch.nn import MSELoss

import numpy as np
import gensim


# remember our multi-layer lstm works like: given an input with shape (N, T, D), it gives a result of shape (N, T, H)
# as the input of the next layer.
# local h matrix is handled in pytorch so we do not need to care about
def train_lstm(sentence, slots, model):
    processed_s = gensim.utils.simple_preprocess(sentence)
    tokenizer = Tokenizer()
    tokenizer.load_gensim_model(model)
    sentence_vec = tokenizer.sentence2vecs(processed_s)
    print(sentence_vec.shape)

    lstm_model = lunet.lstm_net()
    outputs, hidden = lstm_model(sentence_vec)
    lstm_solver = lunet.get_adam_optimizer(lstm_model)

    lstm_model.forward(sentence_vec)
    mse_loss = MSELoss(outputs, slots)

    mse_loss.backward()
    lstm_solver.step()

    lunet.save_checkpoint(lstm_model, lstm_solver, '../../../data/nn/lstm.pth.tar')



