import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy.stats import entropy
import nqueens
import itertools
import curves
import random

steps = 100
theta = np.linspace(0, 2*np.pi, steps)


def series_complex_coeff(c, t, T):
    """calculates the Fourier series with period T at times t,
       from the complex coeff. c"""
    tmp = np.zeros((t.size), dtype=np.complex64)
    for k, ck in enumerate(c):
        # sum from 0 to +N
        tmp += ck * np.exp(2j * np.pi * k * t / T)
        # sum from -N to -1
        if k != 0:
            tmp += ck.conjugate() * np.exp(-2j * np.pi * k * t / T)
    return tmp.real


def build_fun(sol):
    y = []
    for i in range(len(sol)-1):
        s1 = sol[i]
        s2 = sol[i+1]
        xx, yy = curves.half_orbit((s1, 0), (s2, 0))
        y.extend(yy)

    x = np.linspace(0, np.pi*len(sol), len(y))

    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.show()

    return x, y


def build_random_fun(sol):
    y = []
    for i in range(len(sol)-1):
        s1 = sol[i] + random.random() - 0.5
        s2 = sol[i+1] + random.random() - 0.5
        xx, yy = curves.orbit((s1, 0), (s2, 0))
        y.extend(yy)

    x = np.linspace(0, 2*np.pi*len(sol), len(y))

    return x, y

def check_fourier_coff():
    n = 6
    sols = nqueens.n_queens(n).all_solutions

    all_perm = itertools.permutations(sols)

    for i, sol in enumerate(all_perm):
        if i in [3]: #, 9, 11, 12]:
            wave = np.reshape(sol, (-1,)).tolist()
            #wave.append(wave[0])
            wave.insert(0, wave[-1])
            print(wave)

            x, y = build_fun(wave)

            coff = np.fft.rfft(y)

            fig, ax = plt.subplots()
            lim = 100
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.scatter(coff.real, coff.imag, picker = 5)
            plt.show()


check_fourier_coff()

