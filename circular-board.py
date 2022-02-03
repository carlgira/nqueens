import numpy as np
import matplotlib.pyplot as plt
import nqueens
import super_queens as sq
import math
import itertools

import string
digs = string.digits + string.ascii_letters

def int2base(x, base):
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[x % base])
        x = x // base

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)

def get_equation(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1)/(x2 - x1)
    b = y1 - m*x1
    return m, b

def data2binary(data, set_size=-1):

    num_bits = math.floor(math.log2(max(data))) + 1

    if num_bits*len(data) % set_size != 0:
        raise Exception('bad')

    if set_size == -1:
        set_size = num_bits

    l = np.flip(np.reshape(np.unpackbits(np.reshape(data, (-1, 1)), axis=1)[:, -num_bits:], (-1, set_size)), axis=1)
    return [''.join(v.tolist()) for v in l.astype(str)]

def binary2base(data, set_size=-1):
    return int(''.join(data2binary(data, set_size)), 2)

# print(binary2number(np.array([3,2,1], dtype=np.uint8)))

#fig = plt.figure(figsize=(10, 10))
#ax = plt.subplot2grid((1, 1), (0, 0))
#for i, g in enumerate(groups):
#    print(i)
#    b =[binary2number(np.array(v, dtype=np.uint8)) for v in g]

n = 7
sols = nqueens.n_queens(n)
sqi = sq.SuperQueens(n)
isols = sqi.init_sols()
other_sols, _ = sqi.other_sols(isols, sols.all_solutions)
other_sols = [np.array(v) for v in other_sols]
groups = sols.group_solutions()

