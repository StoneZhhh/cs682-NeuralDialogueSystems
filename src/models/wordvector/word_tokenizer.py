import numpy as np
import gzip
import logging

import gensim
from gensim.models import Word2Vec

from nltk.corpus import brown


# logging.basicConfig(format=' % (asctime)s: % (levelname)s: % (message)s', level=logging.INFO)


class Tokenizer:
    def __init__(self, size=10):
        ###########################################################
        # definition of tokenized_sentences:                      #
        # a list of tokenized_sentences.                          #
        # for each tokenzies_sentence, it is a list of tokens     #
        # we use genis word2vec , see more at                     #
        # https://radimrehurek.com/gensim/models/word2vec.html    #
        ###########################################################

        self.tokenized_sentences = []
        self.gensim_model = None
        self.size = size

    # load the corpus and do some preprocessing and return list of words for each sentence
    # preprocessing means that make all words lower-capital and ignore very short words
    # see more at https://radimrehurek.com/gensim/utils.html
    def gensim_load_corpus(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            for i, line in enumerate(f):
                # if i % 10000 == 0:
                #     logging.info("read {0} reviews".format(i))
                self.tokenized_sentences.append(gensim.utils.simple_preprocess(line))

    # train the model using the tokenized sentences in memory
    def gensim_train(self):
        self.gensim_model = Word2Vec(self.tokenized_sentences, size=self.size, window=5, min_count=1, workers=4)

    # train the model using Brown U data
    def gensim_brown_train(self):
        self.gensim_model = Word2Vec(brown.sents(), min_count=1)

    # delete the pretrained model:
    def clean_gensim_model(self):
        self.gensim_model = None

    # load pretrained model:
    def load_gensim_model(self, model_name):
        self.gensim_model = Word2Vec.load(model_name)

    # save the model:
    def save_gensim_model(self, model_name):
        self.gensim_model.save(model_name + '.model')

    # get wordvector using existing genis model
    def word2vec(self, word):
        if word in self.gensim_model.wv:
            return np.array(self.gensim_model.wv[word])
        else:
            return np.zeros(self.size)

    # get wordvectors using existing genis model
    def sentence2vecs(self, sentence):
        # this sentence vec will be a (N, D) in the T th iteration of the whole (N, T, D) T sentences
        vec = np.zeros((len(sentence), self.size))
        for i in range(len(sentence)):
            vec[i, :] = self.word2vec(sentence[i])
        return vec  # (len(sentence), self.size)


# generate models for testing
# tokenizer = Tokenizer()
# tokenizer.gensim_brown_train()
# tokenizer.save_gensim_model('../../../data/word2vec/brown')
