"""
GA sobre un conjunto de pendulos acoplados, con nuestro testigo como fitness.

Genoma: frecuencias w_i, fases iniciales phi_i, acoplamiento K (Kuramoto).
Fenotipo: la permutacion = orden de las fases en el instante T.
Fitness : -(conflictos), que es -(<S^2> - N)/1 salvo constante.

La conexion con lo de hoy: si las w_i estan en progresion aritmetica y K=0, las
fases en T son (w_0 + i*D)*T = alpha*i + beta -- exactamente la rotacion
Sturmiana. Asi que la pregunta interesante no es si el GA encuentra soluciones,
sino SI REDESCUBRE la progresion aritmetica por su cuenta.
"""

import numpy as np


def phases(w, phi, K, T, steps=60):
	"""Kuramoto: dtheta_i/dt = w_i + (K/n) sum_j sin(theta_j - theta_i)."""
	th = phi.copy()
	if K == 0:
		return (th + w * T) % (2 * np.pi)
	dt = T / steps
	for _ in range(steps):
		d = th[None, :] - th[:, None]
		th = th + dt * (w + K * np.sin(d).mean(axis=1))
	return th % (2 * np.pi)


def perm_of(th):
	return np.argsort(np.argsort(th))


def conflicts(p):
	n = len(p); i = np.arange(n)
	return (n - len(set(i + p))) + (n - len(set(i - p)))


def ga(n, K=0.0, pop=300, gens=400, T=1.0, seed=0, elite=.2, mut=.25):
	rng = np.random.default_rng(seed)
	W = rng.uniform(0, 2 * np.pi, (pop, n))
	P = rng.uniform(0, 2 * np.pi, (pop, n))
	best, best_gen, best_ind = 99, -1, None
	for g in range(gens):
		fit = np.array([conflicts(perm_of(phases(W[k], P[k], K, T))) for k in range(pop)])
		o = np.argsort(fit)
		if fit[o[0]] < best:
			best, best_gen, best_ind = int(fit[o[0]]), g, (W[o[0]].copy(), P[o[0]].copy())
			if best == 0: break
		ne = max(2, int(elite * pop))
		W, P = W[o[:ne]], P[o[:ne]]
		iW = rng.integers(0, ne, pop - ne); jW = rng.integers(0, ne, pop - ne)
		m = rng.random((pop - ne, n)) < .5
		cW = np.where(m, W[iW], W[jW]); cP = np.where(m, P[iW], P[jW])
		mm = rng.random((pop - ne, n)) < mut
		cW = np.where(mm, cW + rng.normal(0, .4, cW.shape), cW) % (2 * np.pi)
		cP = np.where(mm, cP + rng.normal(0, .4, cP.shape), cP) % (2 * np.pi)
		W = np.vstack([W, cW]); P = np.vstack([P, cP])
	return best, best_gen, best_ind


def arithmetic_score(w):
	"""¿estan las frecuencias en progresion aritmetica? R^2 de ajustar w ordenadas a una recta."""
	s = np.sort(w); i = np.arange(len(s))
	pred = np.polyval(np.polyfit(i, s, 1), i)
	ss = ((s - s.mean()) ** 2).sum()
	return 1 - ((s - pred) ** 2).sum() / ss if ss > 0 else 0.0


if __name__ == '__main__':
	print(f'{"N":>3} {"K":>5} {"conflictos":>11} {"generacion":>11} {"progr. aritm. R^2":>19}')
	print('-' * 54)
	for n in (8, 10, 12, 14, 16):
		for K in (0.0, 1.5):
			b, g, ind = ga(n, K=K)
			r2 = arithmetic_score(ind[0]) if ind is not None else float('nan')
			print(f'{n:>3} {K:>5.1f} {b:>11} {g:>11} {r2:>19.3f}')
