"""
pp.py -- poda progresiva.  Una regla por vez, y se mide que queda.

Objeto: TODAS las permutaciones de N (las "enumeraciones"). Cada una da una
onda real

	S(x) = sum_k coef[k] * cos(k*x)      coef[k] = #reinas en la diagonal k

con las dos bandas disjuntas (d+ -> armonicos 1..2N-1, d- -> 2N+1..4N-1).
Una permutacion es solucion <=> todos los coef son 0 o 1.

La poda es un embudo: se aplica una regla, se cuenta lo que sobrevive, se
comprueba que NO se ha perdido ninguna solucion, y se pasa a la siguiente.

Toda regla se deriva SOLO de N. La cantera es el conjunto FACTIBLE: los pares
de subconjuntos (A, B) que cumplen los tres momentos invariantes. Se calcula
sin resolver N-reinas y es diminuto (22 pares en N=7, 762 en N=10). Ninguna
regla mira una solucion; si lo hiciera seria una tabla de consulta disfrazada.

	R1  envolvente puntual, bandas SUELTAS   (M0, M1 por banda)
	R2  envolvente puntual, bandas ATADAS    (+ M2, que las acopla)
	R3  reticula de S(pi)                    (valor exacto, O(N))
	R4  energia de la cola                   (un escalar por permutacion)
	Rm  momento espectral Mm = sum k^m c_k   (la m-esima derivada en el origen)

VEREDICTO, medido en N = 7..10 sobre el universo completo:

	R2 se come a R1 y a R3. Su envolvente esta contenida en la de R1 por
	construccion, y evaluada en x = pi reproduce la reticula de R3 sin
	huecos. Cero supervivientes extra en los cuatro N.

	R4 es el testigo disfrazado. <S^2> sobre TODO el periodo es N +
	conflictos exactamente; recortarlo a la cola apenas cambia nada porque
	la cabeza es identica para todas (M0, M1, M2 invariantes). Correlacion
	con el numero de conflictos: 0.995 / 0.991 / 0.984 / 0.982. No es una
	regla nueva, es una version RUIDOSA del testigo exacto.

Asi que de las cuatro primeras solo sobrevive R2. Lo que si suma es la CABEZA
de la curva, que habiamos tirado: M0, M1 y M2 son invariantes -- por eso todas
las curvas salen del origen pegadas -- pero M3 en adelante varia, y acotarlo
por el rango de los factibles quita otro 25% en N=10. Satura rapido:

	R2 24527  +M3 19396  +M4 18976  +M5 18773  +M6 18634  +M7 18548  +M8 18455

RULES = R2 + M3 + M4. ALL_RULES conserva las cuatro originales para poder
reproducir la comparacion.
"""

import numpy as np
from itertools import permutations, islice, combinations
from collections import defaultdict

from diag_waves.bounds import moments, band_target

TAIL = 0.5						# "los valores despues de 0.5"
GRID = 600
NEG = -1e18
EPS = 1e-9


# ------------------------------------------------------------------ el universo

def basis(n, x):
	"""cos(k*x) para k = 0..4N-1."""
	return np.cos(np.outer(np.arange(4 * n), x))


def grid(n, x0=TAIL, m=GRID):
	return np.linspace(x0, np.pi, m)


def coef_chunks(n, size=20000):
	"""coef (bloque, 4N) para todas las permutaciones, por bloques."""
	it = permutations(range(n))
	i = np.arange(n)
	while True:
		block = np.array(list(islice(it, size)), dtype=np.int64)
		if not len(block):
			return
		coef = np.zeros((len(block), 4 * n))
		rows = np.repeat(np.arange(len(block)), n)
		np.add.at(coef, (rows, (i + block).ravel() + 1), 1)
		np.add.at(coef, (rows, (i - block).ravel() + n + 2 * n), 1)
		yield coef


def is_solution(coef):
	"""Solucion <=> ninguna diagonal repetida <=> todos los coeficientes <= 1."""
	return (coef <= 1).all(axis=1)


# ------------------------------------------------------------- la cantera factible

def band_subsets(n):
	"""
	Los subconjuntos que puede ocupar UNA banda de una solucion:

		A subset de {1..2N-1},  |A| = N,  sum(A) = N^2

	(M0 y M1 restringidos a la banda). Diminuto: 94 en N=7, 2934 en N=10.
	"""
	return [np.array(A) for A in combinations(range(1, 2 * n), n) if sum(A) == n * n]


def feasible_coefs(n):
	"""
	Los espectros que sobreviven a los TRES momentos. M2 es el unico que ata las
	dos bandas: sum(A^2) + sum(B^2) = T(N). Devuelve (P, 4N).
	"""
	subs = band_subsets(n)
	by_sq = defaultdict(list)
	for A in subs:
		by_sq[int((A * A).sum())].append(A)
	tgt = band_target(n)
	out = []
	for q, group in by_sq.items():
		for B in by_sq.get(tgt - q, ()):
			for A in group:
				c = np.zeros(4 * n)
				c[A] = 1
				c[2 * n + B] = 1
				out.append(c)
	return np.array(out)


def loose_coefs(n):
	"""Igual pero SIN M2: el producto cartesiano de las dos bandas."""
	subs = band_subsets(n)
	return subs


# ------------------------------------------------------------------- las reglas

def rule_loose(n, x):
	"""R1 -- envolvente puntual con las bandas sueltas.

	Cada banda cumple M0 y M1 por separado, pero no se exige M2. El extremo del
	total es la suma de los extremos de cada banda, asi que basta recorrer los
	subconjuntos de una banda una vez.
	"""
	b = basis(n, x)
	subs = band_subsets(n)
	v1 = np.array([b[A].sum(0) for A in subs])
	v2 = np.array([b[2 * n + A].sum(0) for A in subs])
	hi = v1.max(0) + v2.max(0)
	lo = v1.min(0) + v2.min(0)

	def keep(coef):
		s = coef @ b
		return ((s <= hi + EPS) & (s >= lo - EPS)).all(axis=1)

	keep.curves = (lo, hi)
	return keep


def rule_tight(n, x):
	"""R2 -- envolvente puntual con las bandas atadas por M2.

	Ahora A y B no son libres: sum(A^2) + sum(B^2) esta fijado. El techo baja
	porque el A que maximiza y el B que maximiza rara vez son compatibles.
	"""
	b = basis(n, x)
	s = feasible_coefs(n) @ b
	hi, lo = s.max(0), s.min(0)

	def keep(coef):
		t = coef @ b
		return ((t <= hi + EPS) & (t >= lo - EPS)).all(axis=1)

	keep.curves = (lo, hi)
	return keep


def rule_pi(n, x=None):
	"""R3 -- la reticula de S(pi).

	S(pi) = sum_k (-1)^k coef[k]. Desarrollando los dos desplazamientos:

		S(pi) = (2E - N) * ((-1)^N - 1),   E = #{i : i + perm[i] par}

	N par  -> 0 SIEMPRE, para toda permutacion: no poda nada aqui.
	N impar-> 2(N - 2E), y con E = 2t - 1 queda S(pi) = 2N + 4 - 8t: una
			  reticula de paso 8. Los espectros factibles solo alcanzan
			  ALGUNOS de sus escalones, y esos son los permitidos.
	"""
	allowed = set(np.rint(feasible_coefs(n) @ np.cos(np.arange(4 * n) * np.pi)).astype(int).tolist())
	v = np.cos(np.arange(4 * n) * np.pi)

	def keep(coef):
		s = np.rint(coef @ v).astype(int)
		return np.isin(s, list(allowed))

	keep.allowed = sorted(allowed)
	keep.stat = lambda coef: np.rint(coef @ v).astype(int)
	return keep


def rule_energy(n, x):
	"""R4 -- la energia de la cola.

	<S^2> sobre TODO el periodo es el testigo exacto (N + conflictos), asi que
	usarlo seria hacer trampa: separa soluciones por definicion. Restringido a
	x > 0.5 ya no lo es -- es un escalar distinto, y se acota por el rango que
	alcanzan los espectros factibles.
	"""
	b = basis(n, x)
	e = ((feasible_coefs(n) @ b) ** 2).mean(axis=1)
	lo, hi = e.min(), e.max()

	def keep(coef):
		t = ((coef @ b) ** 2).mean(axis=1)
		return (t >= lo - EPS) & (t <= hi + EPS)

	keep.band = (lo, hi)
	keep.stat = lambda coef: ((coef @ b) ** 2).mean(axis=1)
	return keep


def rule_moment(m):
	"""
	Rm -- el m-esimo momento espectral, acotado por lo que alcanzan los
	factibles.

		Mm = sum_k k^m * c_k = (-1)^(m/2) * la m-esima derivada de la helice en 0

	M0, M1 y M2 son INVARIANTES (mismo valor para toda permutacion): por eso
	todas las curvas salen del origen pegadas. M3 es el primero que varia --
	333 valores distintos en N=7 -- y es donde empieza la informacion que
	tiramos al quedarnos con la cola x > 0.5.
	"""
	def make(n, x=None):
		w = np.arange(4 * n, dtype=float) ** m
		f = feasible_coefs(n) @ w
		lo, hi = f.min(), f.max()

		def keep(coef):
			v = coef @ w
			return (v >= lo - 1e-6) & (v <= hi + 1e-6)

		keep.band = (lo, hi)
		keep.stat = lambda coef: coef @ w
		keep.order = m
		return keep

	make.__name__ = f'rule_M{m}'
	return make


def _channel_basis(n, m=400):
	x = np.linspace(0, 2 * np.pi, m)
	k = np.arange(1, 2 * n)
	return np.exp(1j * np.outer(k, x)).T				# (m, 2N-1)


def rule_invariant(n, x=None, m=400):
	"""
	R5 -- la envolvente del INVARIANTE de simetria.

		P(x) = |W+(x)|^2      M(x) = |W-(x)|^2

	El par NO ORDENADO {P, M} es exactamente invariante bajo el grupo diedrico
	del tablero (ver orbit_geo.py): las 8 soluciones de una orbita dan la misma
	curva. Y P+M y P*M determinan ese par en cada x -- son sus dos funciones
	simetricas -- asi que acotando las dos se exprime todo el invariante.

	No sirve para elegir UN representante por orbita: ninguna funcion invariante
	puede, por definicion, distinguir entre los 8 miembros. Sirve para podar, y
	es lo que mas aporta despues de R2:

		N=10   R2 24527  +M3,M4 18976  +R5 6559     (724 soluciones)

	Y es informacion ORTOGONAL a R2: por separado R5 poda menos (2456 contra
	717 en N=9), pero combinada gana un factor 3. R2 usa la fase (S es la parte
	real); R5 tira la fase pero separa los dos canales.
	"""
	E = _channel_basis(n, m)
	F = feasible_coefs(n)
	Pf = np.abs(F[:, 1:2 * n] @ E.T) ** 2
	Mf = np.abs(F[:, 2 * n + 1:4 * n] @ E.T) ** 2
	band = ((Pf + Mf).min(0), (Pf + Mf).max(0),
			(Pf * Mf).min(0), (Pf * Mf).max(0))

	def keep(coef):
		P = np.abs(coef[:, 1:2 * n] @ E.T) ** 2
		M = np.abs(coef[:, 2 * n + 1:4 * n] @ E.T) ** 2
		s, q = P + M, P * M
		return (((s >= band[0] - 1e-6) & (s <= band[1] + 1e-6)).all(axis=1)
				& ((q >= band[2] - 1e-6) & (q <= band[3] + 1e-6)).all(axis=1))

	keep.band = band
	keep.basis = E
	return keep


ALL_RULES = [
	('R1  envolvente, bandas sueltas', rule_loose),
	('R2  envolvente, bandas atadas (M2)', rule_tight),
	('R3  reticula de S(pi)', rule_pi),
	('R4  energia de la cola', rule_energy),
]

# El embudo que se usa: la envolvente, y luego la cabeza de la curva.
RULES = [
	('R2  envolvente, bandas atadas (M2)', rule_tight),
	('R3  momento M3  (la 3a derivada)', rule_moment(3)),
	('R4  momento M4  (la 4a derivada)', rule_moment(4)),
	('R5  invariante {|W+|,|W-|}', rule_invariant),
]


# ------------------------------------------------------------------- el embudo

def run(n, rules=RULES, x0=TAIL, verbose=True):
	x = grid(n, x0)
	checks = [(name, f(n, x)) for name, f in rules]

	total = 0
	alive = np.zeros(len(checks) + 1, dtype=np.int64)
	kept = np.zeros(len(checks) + 1, dtype=np.int64)
	conf_hist = defaultdict(int)

	for coef in coef_chunks(n):
		sol = is_solution(coef)
		total += len(coef)
		m = np.ones(len(coef), dtype=bool)
		alive[0] += m.sum(); kept[0] += sol.sum()
		for j, (_, keep) in enumerate(checks):
			m &= keep(coef)
			alive[j + 1] += m.sum(); kept[j + 1] += (m & sol).sum()
		c = ((coef ** 2).sum(1) - 2 * n) / 2
		for v, q in zip(*np.unique(c[m].astype(int), return_counts=True)):
			conf_hist[int(v)] += int(q)

	if verbose:
		print(f'N = {n}      x en ({x0}, pi],  {len(x)} muestras,  '
			  f'{len(feasible_coefs(n))} espectros factibles')
		print(f'{"":38} {"quedan":>10} {"%":>7} {"sols":>6}  {"corta":>6}')
		print(f'{"todas las enumeraciones":38} {alive[0]:>10} {100.0:>6.1f}% {kept[0]:>6}')
		for j, (name, _) in enumerate(checks):
			cut = 100 * (alive[j] - alive[j + 1]) / max(alive[j], 1)
			print(f'{name:38} {alive[j+1]:>10} {100*alive[j+1]/total:>6.1f}% {kept[j+1]:>6}  {cut:>5.1f}%'
				  + ('' if kept[j + 1] == kept[0] else '   <-- PIERDE SOLUCIONES'))
		print(f'\nlo que queda, por pares en conflicto:')
		for v in sorted(conf_hist)[:8]:
			print(f'   {v:>3} conflictos : {conf_hist[v]:>8}'
				  + ('   (= soluciones)' if v == 0 else ''))
	return alive, kept, total


def solo(n, rules=ALL_RULES, x0=TAIL):
	"""Cada regla SOLA contra el universo entero: cuanto vale por si misma."""
	x = grid(n, x0)
	checks = [(name, f(n, x)) for name, f in rules]
	total = 0
	alive = np.zeros(len(checks), dtype=np.int64)
	kept = np.zeros(len(checks), dtype=np.int64)
	nsol = 0
	for coef in coef_chunks(n):
		sol = is_solution(coef); total += len(coef); nsol += int(sol.sum())
		for j, (_, keep) in enumerate(checks):
			m = keep(coef)
			alive[j] += m.sum(); kept[j] += (m & sol).sum()
	print(f'N = {n}   cada regla POR SEPARADO sobre las {total} enumeraciones')
	for j, (name, _) in enumerate(checks):
		print(f'{name:38} {alive[j]:>10} {100*alive[j]/total:>6.1f}% {kept[j]:>6}/{nsol}')
	return alive, kept, total


# ------------------------------------------------------------------ las figuras

def _mpl():
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	return plt


GREY, BLUE, ORANGE, PURPLE, GREEN = '#dee2e6', '#1c7ed6', '#e8590c', '#7048e8', '#0ca678'


def _sample(n, keep_fn=None, cap=500):
	"""Devuelve (x_full, curvas eliminadas, curvas que quedan, curvas solucion)."""
	x = grid(n, 0.0)
	b = basis(n, x)
	out, kept, sols = [], [], []
	for coef in coef_chunks(n):
		s = coef @ b
		sol = is_solution(coef)
		m = np.ones(len(coef), bool) if keep_fn is None else keep_fn(coef)
		for dst, sel in ((out, ~m), (kept, m & ~sol), (sols, sol & m)):
			if len(dst) < cap:
				idx = np.flatnonzero(sel)[:cap - len(dst)]
				dst.extend(s[idx])
	return x, np.array(out), np.array(kept), np.array(sols)


def plot_r1(n, path=None):
	"""R1: la envolvente con las bandas sueltas."""
	plt = _mpl()
	path = path or f'diag_waves/fig_pp_r1_n{n}.png'
	xr = grid(n)
	lo, hi = rule_loose(n, xr).curves
	x, out, kept, sols = _sample(n, rule_loose(n, xr))

	fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
	for a, curves, title, col in ((ax[0], out, 'se salen -> fuera', GREY),
								  (ax[1], kept, 'dentro -> siguen', BLUE)):
		if len(curves):
			a.plot(x, curves.T, color=col, lw=.5, alpha=.35)
		a.plot(xr, hi, color=PURPLE, lw=2)
		a.plot(xr, lo, color=PURPLE, lw=2)
		a.axvline(TAIL, color='k', ls=':', lw=1)
		a.set_title(title); a.set_xlabel('x')
	if len(sols):
		ax[1].plot(x, sols.T, color=ORANGE, lw=1.1, alpha=.9)
	ax[0].set_ylabel('S(x)')
	fig.suptitle(f'N={n} — R1: cada banda cumple M0 y M1 por su cuenta '
				 f'(naranja = soluciones)', y=.98)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_r2(n, path=None):
	"""R2: que le hace M2 al techo. Las dos envolventes juntas."""
	plt = _mpl()
	path = path or f'diag_waves/fig_pp_r2_n{n}.png'
	xr = grid(n)
	lo1, hi1 = rule_loose(n, xr).curves
	lo2, hi2 = rule_tight(n, xr).curves
	x, out, kept, sols = _sample(n, rule_tight(n, xr))

	fig, ax = plt.subplots(1, 2, figsize=(15, 5.4), sharey=True)
	ax[0].fill_between(xr, lo1, hi1, color=PURPLE, alpha=.13, lw=0,
					   label='R1  bandas sueltas')
	ax[0].fill_between(xr, lo2, hi2, color=GREEN, alpha=.30, lw=0,
					   label='R2  bandas atadas por M2')
	ax[0].plot(xr, hi1, color=PURPLE, lw=1.4); ax[0].plot(xr, lo1, color=PURPLE, lw=1.4)
	ax[0].plot(xr, hi2, color=GREEN, lw=1.8); ax[0].plot(xr, lo2, color=GREEN, lw=1.8)
	ax[0].legend(loc='lower left', fontsize=9)
	ax[0].set_title('M2 ata las dos bandas y el sobre se cierra')
	ax[0].set_ylabel('S(x)')

	if len(out):
		ax[1].plot(x, out[:250].T, color=GREY, lw=.5, alpha=.45)
	if len(sols):
		ax[1].plot(x, sols[:120].T, color=ORANGE, lw=.6, alpha=.35)
	if len(kept):
		ax[1].plot(x, kept[:120].T, color=BLUE, lw=1.1, alpha=.9)
	ax[1].plot(xr, hi2, color=GREEN, lw=2); ax[1].plot(xr, lo2, color=GREEN, lw=2)
	ax[1].set_title('gris = eliminadas, naranja = soluciones, AZUL = los falsos positivos que quedan')
	for a in ax:
		a.axvline(TAIL, color='k', ls=':', lw=1); a.set_xlabel('x')
	fig.suptitle(f'N={n} — R2: la envolvente sobre los {len(feasible_coefs(n))} '
				 f'espectros factibles', y=.98)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_r3(n, path=None):
	"""R3: la reticula de S(pi)."""
	plt = _mpl()
	path = path or f'diag_waves/fig_pp_r3_n{n}.png'
	keep = rule_pi(n)
	v = np.cos(np.arange(4 * n) * np.pi)
	all_s, sol_s = defaultdict(int), defaultdict(int)
	for coef in coef_chunks(n):
		s = np.rint(coef @ v).astype(int)
		sol = is_solution(coef)
		for a, q in zip(*np.unique(s, return_counts=True)):
			all_s[int(a)] += int(q)
		for a, q in zip(*np.unique(s[sol], return_counts=True)):
			sol_s[int(a)] += int(q)

	ks = sorted(all_s)
	fig, ax = plt.subplots(figsize=(9, 4.6))
	ax.bar(ks, [all_s[k] for k in ks], width=2.6,
		   color=[GREEN if k in keep.allowed else GREY for k in ks],
		   label='permutaciones')
	for k in ks:
		if sol_s.get(k):
			ax.text(k, all_s[k], f'\n{sol_s[k]} sols', ha='center', va='bottom',
					fontsize=9, color=ORANGE)
	ax.set_yscale('log'); ax.set_xticks(ks)
	ax.set_xlabel('S(pi)'); ax.set_ylabel('cuantas permutaciones')
	rule = 'S(pi)=0 SIEMPRE: no poda nada' if n % 2 == 0 else \
		   f'reticula de paso 8; factibles solo en {keep.allowed}'
	ax.set_title(f'N={n} — R3: {rule}')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_r4(n, path=None):
	"""R4: la energia de la cola, coloreada por conflictos reales."""
	plt = _mpl()
	path = path or f'diag_waves/fig_pp_r4_n{n}.png'
	xr = grid(n)
	keep = rule_energy(n, xr)
	lo, hi = keep.band
	byc = defaultdict(list)
	for coef in coef_chunks(n):
		e = keep.stat(coef)
		c = (((coef ** 2).sum(1) - 2 * n) / 2).astype(int)
		for v in np.unique(c):
			if v <= 6:
				byc[int(v)].extend(e[c == v][:4000])

	fig, ax = plt.subplots(figsize=(9.5, 4.8))
	cmap = plt.get_cmap('viridis')
	for v in sorted(byc):
		ax.hist(byc[v], bins=90, histtype='step', lw=1.4,
				color=ORANGE if v == 0 else cmap(v / 7),
				label=f'{v} conflictos' + (' (soluciones)' if v == 0 else ''))
	ax.axvspan(lo, hi, color=GREEN, alpha=.15, lw=0)
	ax.axvline(lo, color=GREEN, lw=1.6); ax.axvline(hi, color=GREEN, lw=1.6)
	ax.set_yscale('log'); ax.legend(fontsize=8)
	ax.set_xlabel('<S^2> sobre x > %.1f' % TAIL); ax.set_ylabel('cuentas')
	ax.set_title(f'N={n} — R4: la energia de la cola es casi un contador de '
				 f'conflictos (+{(np.mean(byc[1])-np.mean(byc[0])):.2f} por conflicto)')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def purity(ns=(7, 8, 9, 10), m=GRID):
	"""R2 sola: cuanto deja, y que fraccion de lo que deja son soluciones."""
	print(f'{"N":>3} {"universo":>10} {"R2 deja":>9} {"sols":>7} {"pureza":>8}')
	rows = []
	for n in ns:
		x = grid(n, m=m)
		keep = rule_tight(n, x)
		tot = surv = nsol = 0
		for coef in coef_chunks(n, 40000):
			sol = is_solution(coef)
			tot += len(coef); nsol += int(sol.sum()); surv += int(keep(coef).sum())
		print(f'{n:>3} {tot:>10} {surv:>9} {nsol:>7} {100*nsol/surv:>7.1f}%')
		rows.append((n, tot, surv, nsol))
	return rows


def sample_rate(ns=range(7, 15), draws=2_000_000, seed=0):
	"""
	Que fraccion del universo deja pasar R2, por muestreo. Los N grandes no se
	pueden enumerar, pero una fraccion del orden del 1% se estima de sobra con
	dos millones de permutaciones al azar.
	"""
	rng = np.random.default_rng(seed)
	out = []
	for n in ns:
		keep = rule_tight(n, grid(n))
		i = np.arange(n)
		surv = tot = 0
		while tot < draws:
			blk = np.array([rng.permutation(n) for _ in range(40000)])
			coef = np.zeros((len(blk), 4 * n))
			rows = np.repeat(np.arange(len(blk)), n)
			np.add.at(coef, (rows, (i + blk).ravel() + 1), 1)
			np.add.at(coef, (rows, (i - blk).ravel() + 3 * n), 1)
			surv += int(keep(coef).sum()); tot += len(blk)
		out.append((n, len(feasible_coefs(n)), surv / tot))
		print(f'N={n:>3}  factibles {out[-1][1]:>7}  R2 deja {100*out[-1][2]:>8.4f}%', flush=True)
	return out


# Q(N)/N!: el suelo real, para comparar con lo que deja R2.
QN = {7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712, 14: 365596}


def plot_decay(rows=None, path='diag_waves/fig_pp_decay.png'):
	"""R2 contra el suelo. La distancia entre las dos curvas es lo que falla."""
	from math import factorial
	plt = _mpl()
	rows = rows or sample_rate()
	ns = [r[0] for r in rows]
	fig, ax = plt.subplots(figsize=(8.4, 5.0))
	ax.semilogy(ns, [r[2] for r in rows], 'o-', color=GREEN, lw=2.2,
				label='lo que deja R2')
	ax.semilogy(ns, [QN[n] / factorial(n) for n in ns], 's--', color=ORANGE, lw=2,
				label='lo que son soluciones  Q(N)/N!')
	ax2 = ax.twinx()
	ax2.semilogy(ns, [r[1] for r in rows], '^:', color=PURPLE, lw=1.4, alpha=.7)
	ax2.set_ylabel('espectros factibles', color=PURPLE)
	ax.set_xlabel('N'); ax.set_ylabel('fraccion de las N! permutaciones')
	ax.legend(loc='lower left'); ax.grid(alpha=.25)
	ax.set_title('R2 es exacta hasta N=8 y se despega despues')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_moment(n, m=3, path=None):
	"""Rm: donde caen las permutaciones y donde pueden caer las soluciones."""
	plt = _mpl()
	path = path or f'diag_waves/fig_pp_m{m}_n{n}.png'
	keep = rule_moment(m)(n)
	lo, hi = keep.band
	allv, solv = [], []
	for coef in coef_chunks(n):
		v = keep.stat(coef); sol = is_solution(coef)
		allv.append(v[::7]); solv.append(v[sol])
	allv = np.concatenate(allv); solv = np.concatenate(solv)

	fig, ax = plt.subplots(figsize=(9.5, 4.8))
	bins = np.linspace(min(allv.min(), lo), max(allv.max(), hi), 160)
	ax.hist(allv, bins=bins, color=GREY, label='permutaciones (1 de cada 7)')
	ax.hist(solv, bins=bins, color=ORANGE, label='soluciones')
	ax.axvspan(lo, hi, color=GREEN, alpha=.15, lw=0)
	for v in (lo, hi):
		ax.axvline(v, color=GREEN, lw=1.8)
	ax.set_yscale('log'); ax.legend(fontsize=9)
	ax.set_xlabel(f'M{m} = sum k^{m} c_k')
	ax.set_ylabel('cuentas')
	frac = 100 * float(((allv >= lo) & (allv <= hi)).mean())
	ax.set_title(f'N={n} — R{m}: banda permitida por los espectros factibles '
				 f'(deja pasar el {frac:.0f}%)')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def peaks(coef, m=2000):
	"""Numero de maximos locales de S(x) sobre un periodo completo."""
	n = coef.shape[1] // 4
	x = 2 * np.pi * np.arange(m) / m
	S = coef @ np.cos(np.outer(np.arange(4 * n), x))
	return ((S > np.roll(S, 1, 1)) & (S >= np.roll(S, -1, 1))).sum(1)


def rule_peaks(n, x=None, m=2000):
	"""
	Rp -- el numero de PICOS de S(x), acotado por el de los factibles.

	S es un polinomio trigonometrico de grado 4N-1, asi que puede tener hasta
	4N-1 maximos. Una solucion gasta 2N armonicos DISTINTOS, todos con
	amplitud 1: es la configuracion mas "rizada" posible. Cada conflicto funde
	dos armonicos en uno de amplitud 2 y alisa la curva.

	Por eso el numero de picos baja de forma monotona con los conflictos... pero
	tan despacio que no sirve para separar:

		N=10  conflictos 0 1 2 3 4 5 6 7  ->  picos medios
					   30.87 30.46 30.19 30.07 29.95 29.86 29.79 29.71

	Las soluciones tienen mas picos que la media, si -- pero por 0.4 sobre un
	rango de 25 a 37. Sola poda el 1%, y detras del embudo no anade nada
	(N=9: 433 -> 433; N=10: 6559 -> 6532).
	"""
	pf = peaks(feasible_coefs(n), m)
	lo, hi = int(pf.min()), int(pf.max())

	def keep(coef):
		p = peaks(coef, m)
		return (p >= lo) & (p <= hi)

	keep.band = (lo, hi)
	keep.stat = lambda coef: peaks(coef, m)
	return keep


def plot_peaks(n=10, path=None, m=2000):
	"""Cuantos picos tienen las soluciones y cuantos el resto."""
	plt = _mpl()
	path = path or f'diag_waves/fig_pp_peaks_n{n}.png'
	keep = rule_peaks(n, m=m)
	lo, hi = keep.band
	byc = defaultdict(list)
	allp, solp = [], []
	for coef in coef_chunks(n):
		p = keep.stat(coef)
		c = (((coef ** 2).sum(1) - 2 * n) / 2).astype(int)
		sol = is_solution(coef)
		allp.append(p); solp.append(p[sol])
		for v in np.unique(c):
			if v <= 8:
				byc[int(v)].append((int(p[c == v].sum()), int((c == v).sum())))
	allp = np.concatenate(allp); solp = np.concatenate(solp)

	fig, ax = plt.subplots(1, 2, figsize=(14, 5))
	bins = np.arange(allp.min() - .5, allp.max() + 1.5)
	ha, _ = np.histogram(allp, bins=bins)
	hs, _ = np.histogram(solp, bins=bins)
	ctr = bins[:-1] + .5
	ax[0].bar(ctr, ha / ha.sum(), width=.85, color=GREY, label='todas las permutaciones')
	ax[0].bar(ctr, hs / hs.sum(), width=.5, color=ORANGE, label='soluciones')
	ax[0].axvline(lo - .5, color=GREEN, lw=2); ax[0].axvline(hi + .5, color=GREEN, lw=2)
	ax[0].axvspan(lo - .5, hi + .5, color=GREEN, alpha=.10, lw=0)
	ax[0].legend(fontsize=9)
	ax[0].set_xlabel('numero de picos de S(x)'); ax[0].set_ylabel('fraccion')
	ax[0].set_title(f'N={n}: media {allp.mean():.2f} contra {solp.mean():.2f}   '
					f'(banda factible [{lo}, {hi}])')

	ks = sorted(byc)
	mu = [sum(a for a, _ in byc[k]) / sum(b for _, b in byc[k]) for k in ks]
	ax[1].plot(ks, mu, 'o-', color=PURPLE, lw=2)
	ax[1].set_xlabel('pares en conflicto'); ax[1].set_ylabel('picos medios')
	ax[1].set_title('cada conflicto funde dos armonicos y alisa la curva\n'
					f'pero solo {mu[0]-mu[-1]:.2f} picos en total')
	ax[1].grid(alpha=.25)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_all(ns=(7, 8, 9), path='diag_waves/fig_pp_all.png', nx=520, ny=420):
	"""
	Todo, sin recortar: el periodo entero y las N! permutaciones.

	Las permutaciones no se dibujan una a una (362 880 curvas en N=9 no caben);
	se acumulan en una densidad exacta -- cada curva pasa por su casilla, no se
	tira ninguna. Encima, todas las soluciones como lineas.
	"""
	plt = _mpl()
	fig, axes = plt.subplots(1, len(ns), figsize=(6.2 * len(ns), 5.4))
	axes = np.atleast_1d(axes)
	for ax, n in zip(axes, ns):
		x = 2 * np.pi * np.arange(nx) / nx
		B = np.cos(np.outer(np.arange(4 * n), x))
		ylim = 2 * n + 1
		ye = np.linspace(-ylim, ylim, ny + 1)
		dens = np.zeros(ny * nx, dtype=np.int64)
		sols = []
		total = 0
		for coef in coef_chunks(n, 20000):
			S = coef @ B
			total += len(coef)
			iy = np.clip(np.searchsorted(ye, S) - 1, 0, ny - 1)
			np.add.at(dens, (iy * nx + np.arange(nx)[None, :]).ravel(), 1)
			m = is_solution(coef)
			if m.any():
				sols.append(S[m])
		dens = dens.reshape(ny, nx)
		sols = np.vstack(sols)

		d = np.log1p(dens)
		ax.imshow(d, origin='lower', aspect='auto', cmap='Blues',
				  vmax=np.percentile(d[d > 0], 97),
				  extent=[0, 2 * np.pi, -ylim, ylim], interpolation='nearest')
		al = float(np.clip(4.0 / np.sqrt(len(sols)), .05, .55))
		ax.plot(x, sols.T, color=ORANGE, lw=.5, alpha=al)
		ax.set_xlabel('x')
		ax.set_title(f'N={n}   {total} permutaciones,  {len(sols)} soluciones')
	axes[0].set_ylabel('S(x)')
	fig.suptitle('azul = densidad de TODAS las permutaciones (escala log), '
				 'naranja = las soluciones', y=.99)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


def plot_funnel(ns=(7, 8, 9, 10), rules=None, path='diag_waves/fig_pp_funnel.png'):
	"""El embudo: que fraccion queda tras cada regla, y donde esta el suelo."""
	plt = _mpl()
	rules = rules or ALL_RULES
	fig, ax = plt.subplots(figsize=(8.6, 5.0))
	cmap = plt.get_cmap('plasma')
	for j, n in enumerate(ns):
		alive, kept, total = run(n, rules=rules, verbose=False)
		ax.plot(range(len(alive)), alive / total, 'o-', color=cmap(j / len(ns)),
				lw=2, label=f'N={n}')
		ax.axhline(kept[0] / total, color=cmap(j / len(ns)), ls=':', lw=1)
	ax.set_yscale('log')
	ax.set_xticks(range(len(rules) + 1))
	ax.set_xticklabels(['todas'] + [r[0].split()[0] for r in rules])
	ax.set_ylabel('fraccion que sobrevive')
	ax.legend(fontsize=9)
	ax.set_title('el embudo (punteado = la fraccion que son soluciones: el suelo)')
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


if __name__ == '__main__':
	import sys
	n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
	run(n); print()
	plot_r2(n); plot_moment(n, 3); plot_moment(n, 4)
	plot_funnel(ns=(8, 9, 10), rules=RULES, path='diag_waves/fig_pp_funnel.png')
