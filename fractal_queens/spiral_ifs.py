r"""
CODIFICACION SIMBOLICA v2 -- los angulos salen de las DIAGONALES.

El problema de sacar el giro de la columna: si giros y traslaciones son ambos
raices N-esimas de la unidad, el IFS conmuta con el grupo ciclico Z_N y el
atractor tiene simetria C_N sea cual sea la permutacion. No codifica nada.

Arreglo: usar los dos indices diagonales, que viven modulo 2N-1:

    d-(k) = k - sigma(k) + N - 1       (diagonal /)   -> DONDE va la copia
    d+(k) = k + sigma(k)               (diagonal \)   -> CUANTO gira la copia

    f_k(z) = r * exp(2 pi i d+(k)/(2N-1)) * z  +  exp(2 pi i d-(k)/(2N-1))

Ahora 2N-1 y N no comparten estructura y la simetria se rompe.

Y el enunciado de las N-reinas se vuelve una propiedad del fractal:

    sigma es solucion  <=>  los N mapas son distintos DOS VECES:
                            N posiciones distintas Y N giros distintos.

Si dos reinas comparten diagonal, dos mapas colapsan al mismo giro (o a la
misma posicion): el fractal pierde una rama y le sale un eje de simetria
espurio. La solucion es exactamente la configuracion de maxima asimetria.
"""
import numpy as np


def diag_maps(sigma, dim=1.7):
	sigma = np.asarray(sigma); N = len(sigma); M = 2 * N - 1
	k = np.arange(N)
	dm = (k - sigma + N - 1)          # diagonal /  en [0, 2N-2]
	dp = (k + sigma)                  # diagonal \  en [0, 2N-2]
	r = N ** (-1.0 / dim)
	lam = r * np.exp(2j * np.pi * dp / M)
	t = np.exp(2j * np.pi * dm / M)
	return lam, t


def witness(sigma):
	"""(#posiciones distintas, #giros distintos). Ambos = N  <=>  sigma es solucion."""
	sigma = np.asarray(sigma); N = len(sigma); k = np.arange(N)
	return len(set((k - sigma).tolist())), len(set((k + sigma).tolist()))


def chaos_game(lam, t, res=1500, walkers=220_000, steps=90, burn=30, seed=0, box=None):
	rng = np.random.default_rng(seed)
	N = len(lam)
	z = 0.3 * (rng.normal(size=walkers) + 1j * rng.normal(size=walkers))
	pts, idx = [], []
	for s in range(steps):
		j = rng.integers(0, N, walkers)
		z = lam[j] * z + t[j]
		if s >= burn:
			pts.append(z.copy()); idx.append(j)
	z = np.concatenate(pts); j = np.concatenate(idx)
	m = np.isfinite(z) & (np.abs(z) < 1e3)
	z, j = z[m], j[m]
	if box is None:
		lo = np.array([z.real.min(), z.imag.min()]); hi = np.array([z.real.max(), z.imag.max()])
		c = (lo + hi) / 2; h = (hi - lo).max() / 2 * 1.03
		box = (c[0] - h, c[0] + h, c[1] - h, c[1] + h)
	ix = np.clip(((z.real - box[0]) / (box[1] - box[0]) * (res - 1)).astype(np.int32), 0, res - 1)
	iy = np.clip(((z.imag - box[2]) / (box[3] - box[2]) * (res - 1)).astype(np.int32), 0, res - 1)
	kk = iy * res + ix
	dens = np.bincount(kk, minlength=res * res).reshape(res, res)
	w = np.exp(2j * np.pi * j / N)
	hx = np.bincount(kk, weights=w.real, minlength=res * res)
	hy = np.bincount(kk, weights=w.imag, minlength=res * res)
	hue = (np.arctan2(hy, hx).reshape(res, res) / (2 * np.pi)) % 1.0
	return dens, hue, box
