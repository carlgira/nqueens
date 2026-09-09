"""
El invariante de simetria como HEURISTICA de poda en el arbol.

La idea: R5 (pp.py) usa el invariante {|W+|, |W-|} sobre un tablero COMPLETO.
Aqui se lleva al arbol de busqueda, sobre tableros a medio llenar.

Primero, lo que NO funciona, con su razon:

	El invariante de lo PUESTO no sirve. AUC 0.56 en N=10, 0.58 en N=11 --
	apenas por encima del azar. Describe las diagonales YA usadas, y estar
	condenado no depende de eso: depende de las que QUEDAN.

	Y su version exacta tampoco. El techo por retardo de la autocorrelacion,
	max A(k) sobre los espectros factibles, resulta ser EXACTO (coincide lag a
	lag con el maximo real de las soluciones en N=10) y aun asi poda el 0.0%
	de los nodos: el maximo lo alcanza alguna solucion, asi que ninguna rama
	condenada lo cruza sin cruzar antes otra cosa.

Lo que si funciona es el invariante de lo PENDIENTE: la autocorrelacion de las
diagonales que todavia estan DISPONIBLES. Es la idea del espectro pendiente,
aplicada al invariante en vez de a la condicion de Hall.

	rasgos                                    AUC N=10   AUC N=11
	la formula de 7 rasgos                      0.906      0.899
	invariante de lo puesto                     0.559      0.575
	invariante de lo PENDIENTE                  0.856      0.852
	formula + puesto                            0.911      0.908
	formula + PENDIENTE                         0.944      0.935
	todo                                        0.944      0.937

El invariante de lo puesto es redundante con la formula; el de lo pendiente
sube el AUC 3.8 puntos con un modelo LINEAL de 29 rasgos.

En poda real (entrenado en N=10 y 11, medido en N=12), a igual conservacion de
soluciones gana entre 7 y 12 puntos de ahorro de nodos. Es una mejora real y
modesta: no cambia el orden de magnitud.
"""

import numpy as np

from diag_waves.prune import harvest, simple_features


def _profile(v, n, bins, npairs):
	d = len(v)
	if d < 2:
		return np.zeros(bins)
	diff = np.abs(np.subtract.outer(v, v))[np.triu_indices(d, 1)]
	return np.histogram(diff / (2 * n - 2), bins=bins, range=(0, 1))[0] / max(npairs, 1)


def placed_features(cur, n, bins=8):
	"""Autocorrelacion de las diagonales YA ocupadas. (No sirve: AUC 0.56.)"""
	d = len(cur)
	dp = np.array([i + cur[i] for i in range(d)])
	dm = np.array([i - cur[i] for i in range(d)])
	npairs = d * (d - 1) // 2
	P = _profile(dp, n, bins, npairs); M = _profile(dm, n, bins, npairs)
	return np.concatenate([P, M, [(P ** 2).sum(), P.max(), (M ** 2).sum(), M.max()]])


def pending_features(cur, n, bins=8):
	"""
	Autocorrelacion de las diagonales aun DISPONIBLES.

	Adimensional (el retardo se normaliza por 2N-2 y se agrupa en `bins`, el
	conteo por el numero de pares), asi que transfiere entre N igual que la
	formula de 7 rasgos.
	"""
	d = len(cur); cols = set(cur)
	dp = {i + cur[i] for i in range(d)}
	dm = {i - cur[i] for i in range(d)}
	ap, am = set(), set()
	for r in range(d, n):
		for c in range(n):
			if c in cols or (r + c) in dp or (r - c) in dm:
				continue
			ap.add(r + c); am.add(r - c)
	ap = np.array(sorted(ap)) if ap else np.zeros(1)
	am = np.array(sorted(am)) if am else np.zeros(1)
	k = max(n - d, 1)
	P = _profile(ap, n, bins, len(ap) * (len(ap) - 1) // 2)
	M = _profile(am, n, bins, len(am) * (len(am) - 1) // 2)
	return np.concatenate([P, M, [len(ap) / (2 * n), len(am) / (2 * n),
								  (P ** 2).sum(), (M ** 2).sum(),
								  len(ap) / k / (2 * n), len(am) / k / (2 * n)]])


def full_features(cur, n):
	return np.concatenate([simple_features(cur, n), pending_features(cur, n)])


# ------------------------------------------------------------------ el modelo

def logistic(F, y, it=1500, lr=.5, l2=1e-3):
	mu, sd = F.mean(0), F.std(0) + 1e-9
	X = np.hstack([(F - mu) / sd, np.ones((len(F), 1))])
	w = np.zeros(X.shape[1])
	for _ in range(it):
		p = 1 / (1 + np.exp(-X @ w))
		w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
	return w, mu, sd


def auc(score, y):
	o = np.argsort(score); y = y[o]
	r = np.arange(1, len(y) + 1)
	n1 = y.sum(); n0 = len(y) - n1
	return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def compare_auc(ns=(10, 11)):
	sets = [('la formula (7)', lambda c, n: simple_features(c, n)),
			('invariante PUESTO', placed_features),
			('invariante PENDIENTE', pending_features),
			('formula + puesto', lambda c, n: np.concatenate([simple_features(c, n), placed_features(c, n)])),
			('formula + PENDIENTE', full_features)]
	for n in ns:
		rows = harvest(n)
		rng = np.random.default_rng(0)
		rows = [rows[i] for i in rng.permutation(len(rows))]
		y = np.array([0. if ok else 1. for _, ok in rows])
		k = int(.7 * len(y))
		print(f'N={n}   {len(y)} estados, {100*y.mean():.1f}% condenados')
		for name, fx in sets:
			F = np.array([fx(c, n) for c, _ in rows])
			w, mu, sd = logistic(F[:k], y[:k])
			s = np.hstack([(F[k:] - mu) / sd, np.ones((len(F) - k, 1))]) @ w
			print(f'   {name:<24} AUC {auc(s, y[k:]):.4f}')


# ------------------------------------------------------------------- la poda

def train(ns=(10, 11), fx=full_features):
	rows = [(c, n, ok) for n in ns for c, ok in harvest(n)]
	F = np.array([fx(c, n) for c, n, _ in rows])
	y = np.array([0. if ok else 1. for *_, ok in rows])
	return logistic(F, y)


def count(n, model=None, fx=full_features, thr=None):
	full = (1 << n) - 1
	dmin = n // 2
	nodes = found = 0
	cur = []
	w, mu, sd = model or (None, None, None)

	def rec(r, cols, dp, dm):
		nonlocal nodes, found
		nodes += 1
		if r == n:
			found += 1; return
		if thr is not None and r >= dmin:
			f = fx(tuple(cur), n)
			if 1 / (1 + np.exp(-(np.append((f - mu) / sd, 1.) @ w))) > thr:
				return
		free = ~(cols | dp | dm) & full
		while free:
			b = free & -free; free ^= b
			cur.append(b.bit_length() - 1)
			rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1)
			cur.pop()

	rec(0, 0, 0, 0)
	return nodes, found


def compare_prune(n=12, thresholds=(.9999, .999, .995, .99, .98, .95, .9)):
	base_fx = lambda c, k: simple_features(c, k)
	Mb = train(fx=base_fx); Mp = train(fx=full_features)
	base, f0 = count(n)
	print(f'N={n}   sin poda {base} nodos, {f0} soluciones')
	print(f'{"umbral":>8} | {"solo formula":>22} | {"formula + pendiente":>22}')
	print(f'{"":>8} | {"nodos":>9} {"ahorro":>6} {"sols":>5} | {"nodos":>9} {"ahorro":>6} {"sols":>5}')
	rows = []
	for t in thresholds:
		a, fa = count(n, Mb, base_fx, t)
		b, fb = count(n, Mp, full_features, t)
		print(f'{t:>8.4f} | {a:>9} {100*(1-a/base):>5.1f}% {100*fa/f0:>4.0f}% |'
			  f' {b:>9} {100*(1-b/base):>5.1f}% {100*fb/f0:>4.0f}%', flush=True)
		rows.append((t, 1 - a / base, fa / f0, 1 - b / base, fb / f0))
	return rows


def plot(rows=None, n=12, path='diag_waves/fig_inv_prune.png'):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	rows = rows or compare_prune(n)
	r = np.array(rows)
	fig, ax = plt.subplots(figsize=(8.4, 5.2))
	ax.plot(100 * r[:, 2], 100 * r[:, 1], 'o-', color='#adb5bd', lw=2,
			label='solo la formula (7 rasgos)')
	ax.plot(100 * r[:, 4], 100 * r[:, 3], 's-', color='#0ca678', lw=2.4,
			label='+ invariante de lo pendiente (29)')
	ax.set_xlabel('soluciones conservadas (%)')
	ax.set_ylabel('nodos ahorrados (%)')
	ax.invert_xaxis(); ax.grid(alpha=.25); ax.legend()
	ax.set_title(f'N={n}: el invariante de lo pendiente sube la curva '
				 f'~10 puntos')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


# --------------------------------------------------- los PICOS como heuristica

def state_coefs(cur, n):
	"""(espectro de lo PUESTO, indicador de lo DISPONIBLE), vectores de 4N."""
	d = len(cur); cols = set(cur)
	dp = {i + cur[i] for i in range(d)}
	dm = {i - cur[i] for i in range(d)}
	a = np.zeros(4 * n); b = np.zeros(4 * n)
	for i in range(d):
		a[i + cur[i] + 1] += 1
		a[2 * n + i - cur[i] + n] += 1
	for r in range(d, n):
		for c in range(n):
			if c in cols or (r + c) in dp or (r - c) in dm:
				continue
			b[r + c + 1] = 1
			b[2 * n + r - c + n] = 1
	return a, b


def peak_features(rows, n, m=512):
	"""
	Los picos que HAY contra los que FALTAN.

	Seis razones adimensionales: picos de la onda de lo puesto, de la onda de
	lo disponible, su cociente, y el deficit hasta el techo factible (absoluto,
	por reina pendiente, y por profundidad).

	Control importante: la profundidad SOLA da AUC 0.51 (azar), asi que el
	0.71 de los picos no es la profundidad disfrazada. Y a profundidad FIJA
	(d = N-2) suben a 0.76 / 0.79: la senal es real.

	Pero ninguna razon por separado vale nada -- las seis estan entre 0.48 y
	0.57. Lo que predice es la COMBINACION, no un cociente.
	"""
	from diag_waves.pp import feasible_coefs
	x = 2 * np.pi * np.arange(m) / m
	B = np.cos(np.outer(np.arange(4 * n), x))

	def npk(S):
		return ((S > np.roll(S, 1, 1)) & (S >= np.roll(S, -1, 1))).sum(1)

	hi = int(npk(feasible_coefs(n) @ B).max())
	A, C, D = [], [], []
	for cur, _ in rows:
		a, b = state_coefs(cur, n)
		A.append(a); C.append(b); D.append(len(cur))
	A = np.array(A); C = np.array(C); D = np.array(D, float)
	P = np.concatenate([npk(A[k:k + 20000] @ B) for k in range(0, len(A), 20000)]).astype(float)
	Q = np.concatenate([npk(C[k:k + 20000] @ B) for k in range(0, len(C), 20000)]).astype(float)
	k = np.maximum(n - D, 1)
	return np.stack([P / (4 * n), Q / (4 * n), P / np.maximum(Q, 1),
					 (hi - P) / k / (4 * n), (hi - P) / (4 * n),
					 P / np.maximum(D, 1) / (4 * n)], 1)


def compare_peaks(ns=(10, 11), path='diag_waves/fig_inv_peaks.png'):
	"""Los picos frente a lo demas, y cuanto sobrevive al combinarlos."""
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	def _auc(F, y, k):
		w, mu, sd = logistic(F[:k], y[:k])
		return auc(np.hstack([(F[k:] - mu) / sd, np.ones((len(F) - k, 1))]) @ w, y[k:])

	labels = ['profundidad sola\n(control)', 'picos (6)', 'autocorr\npendiente (22)',
			  'formula (7)', 'formula\n+ picos', 'formula\n+ autocorr',
			  'todo']
	res = {}
	for n in ns:
		rows = harvest(n)
		rng = np.random.default_rng(0)
		rows = [rows[i] for i in rng.permutation(len(rows))]
		y = np.array([0. if ok else 1. for _, ok in rows]); k = int(.7 * len(y))
		D = np.array([[len(c) / n] for c, _ in rows])
		PK = peak_features(rows, n)
		PD = np.array([pending_features(c, n) for c, _ in rows])
		B7 = np.array([simple_features(c, n) for c, _ in rows])
		res[n] = [_auc(F, y, k) for F in
				  (D, PK, PD, B7, np.hstack([B7, PK]), np.hstack([B7, PD]),
				   np.hstack([B7, PK, PD]))]
		print(f'N={n}  ' + '  '.join(f'{l.splitlines()[0]} {v:.4f}'
									 for l, v in zip(labels, res[n])))

	fig, ax = plt.subplots(figsize=(10.5, 5))
	w = .38
	for j, n in enumerate(ns):
		ax.bar(np.arange(len(labels)) + (j - .5) * w, res[n], w,
			   label=f'N={n}', color=['#0ca678', '#1c7ed6'][j % 2], alpha=.85)
	for j, n in enumerate(ns):
		for i, v in enumerate(res[n]):
			ax.text(i + (j - .5) * w, v + .006, f'{v:.3f}', ha='center', fontsize=7)
	ax.axhline(.5, color='k', ls=':', lw=1)
	ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
	ax.set_ylim(.45, 1.0); ax.set_ylabel('AUC (condenado / vivo)')
	ax.legend(fontsize=9)
	ax.set_title('los picos son senal real, pero la autocorrelacion pendiente '
				 'ya la contiene')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


if __name__ == '__main__':
	compare_auc()
	print()
	plot()
