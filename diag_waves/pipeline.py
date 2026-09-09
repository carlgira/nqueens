"""El pipeline generativo que probamos, dibujado, y dónde muere."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import combinations

from diag_waves.bounds import feasible_pairs

C_GOOD, C_BAD, C_GREY = '#e8590c', '#c92a2a', '#adb5bd'


def flower(n, K, rng):
	M = 2 * n - 1
	th = 2 * np.pi * np.arange(M) / M
	r = np.ones(M)
	for m in range(1, K + 1):
		r += rng.normal(0, 1) * np.cos(m * th + rng.uniform(0, 2 * np.pi))
	return r, tuple(sorted(np.argsort(r)[-n:] + 1))


def realizable(A, b, n):
	A, B = set(A), set(b); col = [False] * n; cur = []
	def rec(r):
		if r == n: return True
		for c in range(n):
			if col[c]: continue
			u, v = r + c + 1, r - c + n
			if u not in A or v not in B: continue
			col[c] = True; A.discard(u); B.discard(v); cur.append(c)
			if rec(r + 1): return True
			col[c] = False; A.add(u); B.add(v); cur.pop()
		return False
	return cur if rec(0) else None


def plot(n=9, path='diag_waves/fig_pipeline.png', seed=3):
	M = 2 * n - 1
	rng = np.random.default_rng(seed)
	tgt = int(round((2/3)*n*(n-1)*(2*n-1) + 2*n*(n-1) + n + n**3))
	fp = feasible_pairs(n)
	A2, b2 = next((A, b) for A, b in fp if realizable(A, b, n))
	perm = realizable(A2, b2, n)

	# la flor que SI pasa el primer momento (siempre es el mismo arco)
	while True:
		r1, A1 = flower(n, 1, rng)
		if sum(A1) == n * n: break

	fig, ax = plt.subplots(2, 4, figsize=(19, 9.5))
	th = 2 * np.pi * np.arange(M) / M
	thc = np.linspace(0, 2 * np.pi, 600)

	def petal(a, r, sel, col, title):
		a.remove()
		p = fig.add_subplot(a.get_subplotspec(), projection='polar')
		V = np.fft.fft(r); fr = np.fft.fftfreq(M, 1/M)
		rc = (V[None] * np.exp(1j*np.outer(thc, fr))).sum(1).real / M
		p.plot(thc, rc, color=col, lw=2)
		m = np.isin(np.arange(1, M+1), sel)
		p.scatter(th[m], r[m], color=col, s=70, zorder=4, edgecolor='k', lw=.6)
		p.scatter(th[~m], r[~m], color=C_GREY, s=35, zorder=4)
		p.set_title(title, fontsize=10, pad=14)
		p.set_yticklabels([]); p.set_xticks(th[::3])
		p.set_xticklabels([str(k+1) for k in range(0, M, 3)], fontsize=7)

	# --- fila 1: el camino de la flor
	petal(ax[0,0], r1, A1, C_BAD, f'1. la flor  $r(\\theta)$, K=1\nse cogen las {n} de mayor radio')
	a = ax[0,1]
	c = np.zeros(M); c[np.array(A1)-1] = 1
	a.stem(np.arange(1, M+1), c, linefmt=C_BAD, markerfmt='o', basefmt=' ')
	a.set_title(f'2. los coeficientes\nA = {A1}\n$\\Sigma A$ = {sum(A1)} = $N^2$  OK', fontsize=9)
	a.set_ylim(0, 1.3); a.set_xlabel('diagonal')

	a = ax[0,2]
	allsets = [S for S in combinations(range(1, M+1), n) if sum(S) == n*n]
	sq = np.array([sum(v*v for v in S) for S in allsets])
	a.hist(sq, bins=40, color=C_GREY, label=f'los {len(allsets)} conjuntos válidos')
	a.axvline(789, color=C_BAD, lw=2.5, label='lo único que da la flor: 789')
	a.axvline(tgt-789, color='#1c7ed6', lw=2, ls='--', label=f'haría falta: {tgt-789}')
	a.set_xlabel('$\\Sigma A^2$'); a.set_ylabel('cuántos')
	a.set_title('3. el filtro del 2º momento', fontsize=10); a.legend(fontsize=7.5)

	a = ax[0,3]; a.axis('off')
	a.text(.5, .55, 'MUERE AQUÍ', ha='center', fontsize=22, color=C_BAD, fontweight='bold')
	a.text(.5, .34, '10 274 125 intentos\n35 353 pasaron los 1ᵒˢ momentos\n'
					'0 pasaron el 2º', ha='center', fontsize=11)
	a.text(.5, .12, 'la flor K=1 solo sabe hacer arcos contiguos,\n'
					'y solo un arco cumple $\\Sigma A=N^2$: siempre el mismo',
		   ha='center', fontsize=9, style='italic', color='#495057')

	# --- fila 2: el camino que sí funciona
	r2 = np.zeros(M); r2[np.array(A2)-1] = 1
	petal(ax[1,0], r2, A2, C_GOOD, '1\'. muestrear un espectro factible\n(no de una curva: de la lista)')
	a = ax[1,1]
	a.stem(np.arange(1, M+1), r2, linefmt=C_GOOD, markerfmt='o', basefmt=' ')
	a.set_title(f'2\'. coeficientes\nA = {A2}\n$\\Sigma A^2$ = {sum(v*v for v in A2)}  OK', fontsize=9)
	a.set_ylim(0, 1.3); a.set_xlabel('diagonal')

	a = ax[1,2]; a.remove()
	a = fig.add_subplot(2, 4, 7, projection='3d')
	coef = np.zeros(4*n)
	for v in A2: coef[v] = 1
	for v in b2: coef[2*n+v] = 1
	x = np.linspace(0, 2*np.pi, 1200)
	Z = coef @ np.exp(1j*np.outer(np.arange(4*n), x))
	a.plot(x, Z.real, Z.imag, color=C_GOOD, lw=1)
	a.set_title('3\'. la hélice', fontsize=10)
	a.set_xticks([]); a.set_yticks([]); a.set_zticks([])

	a = ax[1,3]
	brd = np.zeros((n, n))
	for r_, c_ in enumerate(perm): brd[r_, c_] = 1
	a.imshow(brd, cmap=plt.cm.colors.ListedColormap(['#f8f9fa', C_GOOD]))
	a.set_xticks(range(n)); a.set_yticks(range(n)); a.tick_params(labelsize=7)
	a.set_title(f'4\'. el tablero\n{perm}\n(58% de los factibles llegan aquí)', fontsize=9)

	fig.suptitle(f'N={n}: el pipeline generativo — arriba el que probaste, abajo el que funciona',
				 fontsize=14)
	fig.tight_layout(rect=(0, 0, 1, .95))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot())
