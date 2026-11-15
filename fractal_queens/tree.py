r"""
EL ARBOL DE BACKTRACKING COMO FRACTAL.

Todas las Q(N) soluciones de un N fijo en UN objeto.

Cada colocacion parcial (s_0,...,s_{d-1}) sin conflictos es un nodo. Se le
asigna un angulo por su direccion en base N:

    x = sum_i s_i / N^{i+1}        theta = 2 pi (x + 1/(2 N^d))
    r = d

El radio es la profundidad y el angulo es el prefijo. Las hojas a profundidad N
son las soluciones, y su conjunto es el Cantor de las N-reinas dentro de [0,1):
Q(N) celdas de tamano N^-N, o sea dimension de caja

    dim = log Q(N) / (N log N)

Las ramas que mueren antes de N son la poda, y son la mayor parte del arbol.
"""
import numpy as np


def build(N):
	"""DFS con mascaras de bits. Devuelve nodos, aristas y quien sobrevive."""
	depth = [0]; theta = [0.0]; parent = [-1]; alive = [False]
	stack = [(0, 0, 0, 0, 0, 0.0)]        # (idx, d, cols, dia1, dia2, x)
	full = (1 << N) - 1
	while stack:
		idx, d, cols, d1, d2, x = stack.pop()
		if d == N:
			alive[idx] = True
			p = parent[idx]
			while p >= 0 and not alive[p]:
				alive[p] = True; p = parent[p]
			continue
		free = full & ~(cols | d1 | d2)
		while free:
			b = free & -free; free ^= b
			v = b.bit_length() - 1
			nx = x + v / N ** (d + 1)
			depth.append(d + 1)
			theta.append(2 * np.pi * (nx + 0.5 / N ** (d + 1)))
			parent.append(idx); alive.append(False)
			stack.append((len(depth) - 1, d + 1, cols | b,
			              ((d1 | b) << 1) & full, (d2 | b) >> 1, nx))
	return (np.array(depth), np.array(theta), np.array(parent), np.array(alive))


def profile(N):
	d, th, p, al = build(N)
	W = np.bincount(d, minlength=N + 1)
	L = np.bincount(d[al], minlength=N + 1)
	return W, L, len(d)
