import numpy as np
import math


def dac(n, set_size, data):
    num_bits = math.floor(math.log2(n-1)) + 1

    if num_bits*len(data) % set_size != 0:
        return []

    return np.reshape(np.packbits(np.flip(np.reshape(np.unpackbits(np.reshape(data, (-1, 1)), axis=1)[:, -num_bits:], (-1, set_size)), axis=1), bitorder='little', axis=-1), -1)


n = 8
ll = np.array(list(range(n)), dtype=np.uint8)
print(dac(n, 6, ll))




