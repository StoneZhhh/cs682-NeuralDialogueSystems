import json
import pickle


def load_as_list(path):
    with open(path) as f:
        lines = f.read().splitlines()
    return lines


def read_from_pickle(path):
    data = []
    with open(path, 'rb') as file:
        while True:
            try:
                # encoding of pickle on py3 has some bugs and requires latin1 form
                data.append(pickle.load(file, encoding='latin1'))
            except EOFError:
                break
    if len(data) == 1:
        return data[0]
    return data


# data = read_from_pickle('../../data/dicts.v3.p')
# print(data)
# data = read_from_pickle('../../data/movie_kb.1k.p')
# print(list(data[1].values()))
# print(list(data[1].keys()))
