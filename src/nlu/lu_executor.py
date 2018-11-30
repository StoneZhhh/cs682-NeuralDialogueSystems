import nlu.lu_torch_net as lunet
from models.wordvector.tokenizer import Tokenizer

import torch
import torch.nn as nn

import numpy as np
import gensim


def get_slots(sentence, model):
    processed_s = gensim.utils.simple_preprocess(sentence)
    Tokenizer.load_gensim_model(model)
    sentence_vec = Tokenizer.sentence2vecs(processed_s)


