import numpy as np


def mix_vertical(sols):
    return np.array(sols).transpose().tolist()


def demix_vertical(sols, shape):
    s = np.reshape(sols, shape)
    return np.array(s).transpose().tolist()


s = [[1, 2, 3, 4, 7], [5, 6, 7, 1, 2]]
ss = mix_vertical(s)
sss = demix_vertical(ss, (5, 2))
print(ss, sss)
