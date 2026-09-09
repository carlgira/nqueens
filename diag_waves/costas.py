"""
Costas con la misma maquinaria: la onda 2D y su testigo.

N-reinas:  testigo = <|W|^2> sobre la onda 1D del espectro diagonal
Costas  :  testigo = <|W|^4> sobre la onda 2D del tablero

Porque |W(u,v)|^2 es la transformada de la AUTOCORRELACION 2D A(dx,dy), y la
condicion de Costas es exactamente A(dx,dy) <= 1 fuera del origen. Por Parseval,
<|W|^4> = (1/M^2) sum A(dx,dy)^2, minimo justo cuando todas las A valen 0 o 1.
"""

import numpy as np


def wave2d(p, m=None):
	"""W(u,v) = sum_i exp(2*pi*i*(i*u + p(i)*v)/m)"""
	n = len(p); m = m or (2 * n)
	board = np.zeros((m, m))
	board[np.arange(n), np.asarray(p)] = 1
	return np.fft.fft2(board)


def witness(p, m=None):
	"""<|W|^4>.  Minimo = 2N^2 - N  <=>  es Costas."""
	W = wave2d(p, m)
	return float((np.abs(W) ** 4).mean())


def autocorr(p):
	"""A(dx,dy) = nº de pares con ese desplazamiento."""
	n = len(p); A = {}
	for i in range(n):
		for j in range(n):
			if i == j: continue
			A[(j - i, p[j] - p[i])] = A.get((j - i, p[j] - p[i]), 0) + 1
	return A


def is_costas(p):
	A = autocorr(p)
	return all(v == 1 for v in A.values())


def conflicts(p):
	"""pares de pares que comparten desplazamiento"""
	A = autocorr(p)
	return sum(v * (v - 1) // 2 for v in A.values())
