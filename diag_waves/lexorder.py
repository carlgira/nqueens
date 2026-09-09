"""Las soluciones en orden lexicografico: ¿hay patron?"""

import numpy as np
from math import factorial


def lex_index(p):
	n = len(p); idx = 0
	for i in range(n):
		c = sum(1 for j in range(i + 1, n) if p[j] < p[i])
		idx += c * factorial(n - 1 - i)
	return idx


def sols(n):
	out, cur, full = [], [], (1 << n) - 1
	def rec(r, cols, dp, dm):
		if r == n: out.append(tuple(cur)); return
		f = ~(cols | dp | dm) & full
		while f:
			b = f & -f; f ^= b; c = b.bit_length() - 1
			cur.append(c); rec(r + 1, cols | b, (dp | b) << 1, (dm | b) >> 1); cur.pop()
	rec(0, 0, 0, 0)
	return out


def plot(n=10, path='diag_waves/fig_lexorder.png'):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	S = sols(n)
	S.sort(key=lex_index)
	I = np.array([lex_index(p) for p in S]); T = factorial(n)
	r = np.arange(1, len(I) + 1)
	C1, C2 = '#e8590c', '#1c7ed6'

	fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))

	a = ax[0, 0]
	a.plot(r, I / T, '.', color=C1, ms=3)
	a.plot(r, r / (len(I) + 1), 'k--', lw=1.5, label='reparto uniforme')
	a.set_xlabel('rango de la solución'); a.set_ylabel('índice lexicográfico / N!')
	a.set_title(f'N={n}: las {len(I)} soluciones en orden lexicográfico\n'
				'casi una recta, pero no exacta', fontsize=11)
	a.legend(fontsize=9)

	a = ax[0, 1]
	a.plot(r, I / T - r / (len(I) + 1), color=C1, lw=1)
	a.axhline(0, color='k', lw=.8)
	a.set_xlabel('rango'); a.set_ylabel('desviación de la recta')
	a.set_title(f'El residuo: ondula, no es ruido\nmáx {np.abs(I/T - r/(len(I)+1)).max():.3f}',
				fontsize=11)

	a = ax[1, 0]
	g = np.diff(I) / (T / len(I))
	a.hist(g, bins=np.linspace(0, 6, 60), density=True, color=C1, label='huecos reales')
	xx = np.linspace(0, 6, 200)
	a.plot(xx, np.exp(-xx), 'k--', lw=2, label='exponencial (reparto al azar)')
	a.set_yscale('log'); a.set_xlabel('hueco / hueco medio'); a.set_ylabel('densidad')
	a.set_title(f'Se agrupan: {(g<0.1).mean():.0%} de huecos diminutos\n'
				f'(al azar serían 9.5%)  ·  sd = {g.std():.2f} vs 1.0', fontsize=11)
	a.legend(fontsize=9)

	a = ax[1, 1]
	for k, col in ((1, '#495057'), (2, C2), (3, C1)):
		pref = {}
		for p in S: pref[p[:k]] = pref.get(p[:k], 0) + 1
		v = np.array(sorted(pref.values(), reverse=True))
		a.plot(np.arange(1, len(v) + 1) / len(v), np.cumsum(v) / v.sum(), lw=2,
			   color=col, label=f'primeras {k} filas fijadas ({len(v)} bloques)')
	a.plot([0, 1], [0, 1], 'k:', lw=1.2, label='reparto uniforme')
	a.set_xlabel('fracción de bloques (de mayor a menor)')
	a.set_ylabel('fracción de soluciones acumulada')
	a.set_title('De dónde viene el agrupamiento:\nunos prefijos llevan muchas soluciones y otros ninguna',
				fontsize=11)
	a.legend(fontsize=8.5)

	fig.suptitle('Las soluciones en el orden del espacio de búsqueda', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .94))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot(10))
