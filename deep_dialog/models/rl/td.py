import numpy as np
import itertools


def vw(w, s, f_order):
    return w.dot(fourier_phi(s, f_order))[0][0]


def dvwdw(w, s, f_order):
    return fourier_phi(s, f_order)


def fourier_phi(s, f_order):
    normalized_s = np.array([normalize(s)]).T
    iter = itertools.product(range(f_order + 1), repeat=len(s))
    c = np.array([list(map(int, x)) for x in iter])  # ((n+1)^d, d) = (256, 4) if f_order = 3, d = 4
    return np.cos(np.pi * c.dot(normalized_s))  # ((n+1)^d) = (256, ) if f_order = 3, d = 4