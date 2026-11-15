"""
CODIFICACION GEOMETRICA.

Las N reinas se meten en el plano complejo como las RAICES de un polinomio:

    z_k = (k - c) / N  +  i (sigma(k) - c) / N          c = (N-1)/2
    p(z) = prod_k (z - z_k)

y se dibuja el fractal de Newton relajado:

    F(z) = z - a * p(z)/p'(z) = z - a / sum_k 1/(z - z_k)

Con a=1 son las cuencas clasicas. Con a complejo el punto fijo de cada raiz
pasa a tener multiplicador complejo (1-a), y TODA la frontera se enrosca:
las cuencas entran en cada raiz girando. La espiral es logaritmica con paso
arg(1-a)/log|1-a|.

La solucion no se ve, pero esta ahi entera: los centros de las cuencas SON
las reinas. Se decodifica buscando los atractores.
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def newton_field(roots, a, box, res, iters=90, tol=1e-10):
	x = np.linspace(box[0], box[1], res)
	y = np.linspace(box[2], box[3], res)
	Z = (x[None, :] + 1j * y[:, None]).astype(np.complex128)
	done = np.zeros(Z.shape, bool)
	tim = np.full(Z.shape, iters, np.float64)
	for n in range(iters):
		s = np.zeros_like(Z)
		for r in roots:
			d = Z - r
			np.divide(1.0, d, out=d, where=(d != 0))
			s += d
		step = np.divide(a, s, out=np.zeros_like(s), where=(s != 0))
		Z = Z - step
		near = (~done) & (np.abs(step) < tol)
		tim[near] = n + 1.0
		done |= near
		if done.all():
			break
	d = np.abs(Z[..., None] - roots[None, None, :])
	idx = d.argmin(-1)
	return idx, tim, np.abs(Z[..., None] - roots[None, None, :]).min(-1)


def colorize(idx, tim, nroot, sat=0.72):
	h = (idx * 0.6180339887498949) % 1.0                      # aureo: colores separados
	t = np.log1p(tim); t = (t - t.min()) / (np.ptp(t) + 1e-12)
	v = 0.16 + 0.92 * (1 - t) ** 1.35
	s = sat * (0.35 + 0.65 * t ** 0.6)
	return hsv_to_rgb(np.dstack([h, np.clip(s, 0, 1), np.clip(v, 0, 1)]))


def roots_of(sigma):
	sigma = np.asarray(sigma, float); N = len(sigma); c = (N - 1) / 2
	return ((np.arange(N) - c) + 1j * (sigma - c)) / N
