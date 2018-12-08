import time, os
from deep_dialog.tools.utils import *
import numpy as np


class BaseWrapper:
    def __init__(self, input_size, hidden_size, output_size):
        pass

    # def get_struct(self):
    #     return {'model': self.model, 'update': self.update, 'regularize': self.regularize}

    """ Forward & Backward Function"""

    def one_batch_train(self, Xs, Y):
        pass

    """ Batch Forward & Backward Pass"""

    def batch_train(self, ds, batch, params, predict_mode=False):
        for i, x in enumerate(batch):
            labels = np.array(x['tags_rep'], dtype=int)
            self.one_batch_train(x, labels)

    """ Evaluate on the dataset[split] """

    def eval(self, ds, split, params):
        acc = 0
        total = 0

        total_cost = 0.0
        smooth_cost = 1e-15

        if split == 'test':
            res_filename = 'res_%s_[%s].txt' % (params['model'], time.time())
            res_filepath = os.path.join(params['test_res_dir'], res_filename)
            res = open(res_filepath, 'w')
            inverse_tag_dict = {ds.data['tag_set'][k]: k for k in ds.data['tag_set'].keys()}

        for i, ele in enumerate(ds.split[split]):

            # Ys, cache = self.fwdPass(ele, params, predict_model=True)
            #
            # maxes = np.amax(Ys, axis=1, keepdims=True)
            # e = np.exp(Ys - maxes)  # for numerical stability shift into good numerical range
            # probs = e / np.sum(e, axis=1, keepdims=True)
            #
            labels = np.array(ele['tags_rep'], dtype=int)
            #
            # if np.all(np.isnan(probs)): probs = np.zeros(probs.shape)

            # loss_cost = 0
            # loss_cost += -np.sum(np.log(smooth_cost + probs[range(len(labels)), labels]))
            # total_cost += loss_cost

            Ys, loss_cost = self.one_batch_train(ele, labels)
            total_cost += loss_cost

            maxes = np.amax(Ys, axis=1, keepdims=True)
            e = np.exp(Ys - maxes)  # for numerical stability shift into good numerical range
            probs = e / np.sum(e, axis=1, keepdims=True)

            pred_words_indices = np.nanargmax(probs, axis=1)

            tokens = ele['raw_seq']
            real_tags = ele['tag_seq']
            for index, l in enumerate(labels):
                if pred_words_indices[index] == l: acc += 1

                if split == 'test':
                    res.write('%s %s %s %s\n' % (
                    tokens[index], 'NA', real_tags[index], inverse_tag_dict[pred_words_indices[index]]))
            if split == 'test': res.write('\n')
            total += len(labels)

        total_cost /= len(ds.split[split])
        accuracy = 0 if total == 0 else float(acc) / total

        # print ("total_cost: %s, accuracy: %s" % (total_cost, accuracy))
        result = {'cost': total_cost, 'accuracy': accuracy}
        return result
