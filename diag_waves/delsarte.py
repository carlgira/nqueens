"""
La cota de entropia sobre Q(N): contar sin construir.

Toda la sesion ha preguntado "como son las soluciones". Esto pregunta "cuantas
puede haber", y lo responde sin construir ninguna.

	log Q(N) = H(tablero aleatorio uniforme entre las soluciones)
			 <= sum_ij H(casilla_ij)                    (subaditividad)
			 <= max { sum_ij h(p_ij) }                  sobre el politopo

con h la entropia binaria y el politopo definido por lo que TODA solucion
cumple:

	sum_j p_ij = 1        una reina por fila
	sum_i p_ij = 1        una por columna
	sum_{i+j=d} p_ij <= 1 una por diagonal ascendente
	sum_{i-j=d} p_ij <= 1 una por diagonal descendente

Es un programa convexo (maximo de una funcion concava sobre un politopo), y su
dual se resuelve con gradiente. Estacionariedad da la forma exponencial:

	p_ij = sigma( -(lambda_i + mu_j + alpha_{i+j} + beta_{i-j}) )

que es la misma familia Fermi-Dirac que profile.py encontro ajustando el perfil
de ocupacion. Aqui aparece derivada, no ajustada.

El dual que se minimiza:

	g = sum_ij log(1 + exp(-s_ij)) + sum lambda + sum mu + sum alpha + sum beta
	s_ij = lambda_i + mu_j + alpha_{i+j} + beta_{i-j},   alpha, beta >= 0

y g* es una cota superior RIGUROSA de log Q(N) para todo N, calculable en
O(N^2) por iteracion sin resolver nada combinatorio.
"""

import numpy as np

# A000170: Q(N) para N = 1..27
QN = [1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596,
	  2279184, 14772512, 95815104, 666090624, 4968057848, 39029188884,
	  314666222712, 2691008701644, 24233937684440, 227514171973736,
	  2207893435808352, 22317699616364044, 234907967154122528]


def _solve1d(rest, target, lo=-60.0, hi=60.0, nonneg=False, it=60):
	"""Resuelve sum_j sigma(-(t + rest_j)) = target en t, por biseccion.
	Monotona decreciente en t, asi que la biseccion es segura."""
	for _ in range(it):
		mid = .5 * (lo + hi)
		if (1.0 / (1.0 + np.exp(mid + rest))).sum() > target:
			lo = mid
		else:
			hi = mid
	t = .5 * (lo + hi)
	return max(t, 0.0) if nonneg else t


def solve(n, sweeps=200, diagonals=True):
	"""
	Minimiza el dual por descenso coordenado EXACTO: cada multiplicador se
	resuelve por biseccion dejando los demas fijos. Converge de verdad.

	diagonals=False deja solo filas y columnas -> la cota de permutacion, para
	comparar y ver cuanto aportan las diagonales.
	"""
	i, j = np.indices((n, n))
	dp = (i + j).ravel()
	dm = (i - j + n - 1).ravel()
	lam = np.zeros(n); mu = np.zeros(n)
	al = np.zeros(2 * n - 1); be = np.zeros(2 * n - 1)
	for _ in range(sweeps):
		for r in range(n):								# filas
			rest = mu + al[r + np.arange(n)] + be[r - np.arange(n) + n - 1]
			lam[r] = _solve1d(rest, 1.0)
		for c in range(n):								# columnas
			rest = lam + al[np.arange(n) + c] + be[np.arange(n) - c + n - 1]
			mu[c] = _solve1d(rest, 1.0)
		if not diagonals:
			continue
		for d in range(2 * n - 1):						# diagonal ascendente
			rr = np.arange(max(0, d - n + 1), min(n, d + 1))
			cc = d - rr
			rest = lam[rr] + mu[cc] + be[rr - cc + n - 1]
			al[d] = _solve1d(rest, 1.0, nonneg=True)
		for d in range(2 * n - 1):						# diagonal descendente
			rr = np.arange(max(0, d - n + 1), min(n, d + 1))
			cc = rr - d + n - 1
			rest = lam[rr] + mu[cc] + al[rr + cc]
			be[d] = _solve1d(rest, 1.0, nonneg=True)
	s = lam[i] + mu[j] + al[i + j] + be[i - j + n - 1]
	bound = float(np.logaddexp(0, -s).sum() + lam.sum() + mu.sum() + al.sum() + be.sum())
	return bound, 1.0 / (1.0 + np.exp(s))


def report(ns=(8, 10, 12, 14, 16, 20, 24, 27)):
	"""
	Se compara todo en la escala alpha:  X = (N * e^-alpha)^N.

		alpha real (Simkin)          1.944
		alpha de N!  (Stirling)      1.000
		alpha de la cota             lo que salga; mas alto = mejor
	"""
	print(f'{"N":>3} {"alpha real":>11} {"alpha N!":>10} {"solo fil/col":>13} '
		  f'{"+ diagonales":>13} {"gana":>7}')
	for n in ns:
		b1, _ = solve(n, diagonals=False)
		b2, _ = solve(n, diagonals=True)
		real = np.log(QN[n - 1])
		fact = float(np.sum(np.log(np.arange(1, n + 1))))
		a = lambda L: np.log(n) - L / n
		print(f'{n:>3} {a(real):>11.4f} {a(fact):>10.4f} {a(b1):>13.4f} '
			  f'{a(b2):>13.4f} {a(b2)-a(b1):>7.4f}')
	print('\nSimkin (2021): log Q(N) = N*(log N - 1.944 + o(1)).')
	print('Coste real de las diagonales: 1.944 - 1.000 = 0.944 por reina.')


if __name__ == '__main__':
	report()
