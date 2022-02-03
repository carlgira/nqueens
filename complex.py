#Complex Sinus function  with coloring based to imaginary part
# Based on this comment http://stackoverflow.com/a/6543777
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

import nqueens

#mpl.use('TkAgg')
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import math

n = 7
phi = (1 + math.sqrt(5))/2
spiral1 = lambda c, t, l: math.pow(c, t - 12*math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12*math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2*math.pi + l))
spiral2_inv = lambda c, r, l: 2*math.pi - math.log(r, c) - l


c1 = math.pow(phi, 2/(math.pi))
f = [(2*i)/n for i in range(n)]

a = [(math.pi*2*i)/n for i in range(n)]
r = [spiral1(c1, 10*math.pi + math.pi*2/n, aa) for aa in a]

board = []

# Complex Circular board
m = (np.outer(r, np.cos(a)) + 1j*np.outer(r, np.sin(a))).T

def complex_sol_mult(sol):
    csol = complex_sol(sol)
    return np.array([csol[i+1]/csol[i] for i in range(len(sol)-1)])

def complex_sol(sol):
    return np.array([m[i][v] for i, v in enumerate(sol)])

def complex_2_sol(init_point, csolm):
    csol = [init_point]
    for c in csolm:
        csol.append(csol[-1]*c)
    return np.argmax(np.sum([np.isclose(m, v, 1e-10) for v in csol], axis=1), axis=1)

sols = nqueens.n_queens(n)

for sol in sols.all_solutions:
        csol = complex_sol_mult(sol)
        asol = complex_2_sol(m[0][sol[0]], csol)
        print(csol)
        plt.scatter(csol.real, csol.imag)


plt.show()
