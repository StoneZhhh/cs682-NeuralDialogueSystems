import copy

import deep_dialog.nlu.lu_torch_net as lunet
from deep_dialog.models.wordvector.word_tokenizer import Tokenizer
import deep_dialog.tools.file_loader as fl

import torch
import torch.nn as nn
from torch.nn import MSELoss

import numpy as np
import gensim
import pickle


# remember our multi-layer lstm works like: given an input with shape (N, T, D), it gives a result of shape (N, T, H)
# as the input of the next layer.
# local h matrix is handled in pytorch so we do not need to care about

movie_corpus = fl.read_from_pickle('../../data/movie_kb.1k.p')
t = Tokenizer()
t.load_gensim_model('../../data/word2vec/brown.model')
sentence_vec = t.sentence2vecs(list(movie_corpus[0].values()))
print(sentence_vec)
print(sentence_vec.shape)

# to-do: train the word2vec model using dict.v3.p and make sure we can get a meaningful output


class NLU:
    def __init__(self, filepath):
        """
        :param filepath:
        Load the params from model file. Require model class to store all the content
        about word_dict, slot_dict and so on
        """

        model_params = pickle.load(open(filepath, 'rb'))

        self.word_dict = copy.deepcopy(model_params['word_dict'])
        self.slot_dict = copy.deepcopy(model_params['slot_dict'])
        self.act_dict = copy.deepcopy(model_params['act_dict'])
        self.tag_set = copy.deepcopy(model_params['tag_set'])
        self.params = copy.deepcopy(model_params['params'])
        self.inverse_tag_dict = {self.tag_set[k]: k for k in self.tag_set.keys()}

    def generate_dia_act(self, annotation):
        if len(annotation) == 0:
            pass
        p_annot = annotation.strip('.').strip('?').strip(',').strip('!')

    def parse_str_to_vector(self, string):
        tmp = 'BOS ' + string + ' EOS'
        words = tmp.lower().split(' ')

        vecs = np.zeros((len(words), len(self.word_dict)))
        for w_index, w in enumerate(words):
            if w.endswith(',') or w.endswith('?'): w = w[0:-1]
            if w in self.word_dict.keys():
                vecs[w_index][self.word_dict[w]] = 1
            else:
                vecs[w_index][self.word_dict['unk']] = 1
        rep = {'word_vectors': vecs, 'raw_seq': string}
        return rep
