def mergeDicts(d0, d1):
    """ for all k in d0, d0 += d1 . d's are dictionaries of key -> numpy array """
    for k in d1:
        if k in d0: d0[k] += d1[k]
        else: d0[k] = d1[k]