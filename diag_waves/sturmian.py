"""
La familia Sturmiana:  x_i = (alpha*i + beta) mod 1,  perm = orden de la orbita.

Entropia CERO -- el sistema menos caotico que no se repite -- y sin embargo es
el mejor generador de soluciones que hemos encontrado. La rejilla satura en
G~600: la familia es finita y diminuta (598 permutaciones para N=13, de 6.2e9).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_GOOD, C_BAD = '#e8590c', '#dee2e6'


def grid(n, G):
	i = np.arange(n)
	g = np.linspace(0, 1, G, endpoint=False) + .5 / G
	V = (g[:, None, None] * i + g[None, :, None]) % 1.0
	P = np.argsort(np.argsort(V, axis=-1), axis=-1).astype(np.int8)
	ok = np.ones((G, G), bool)
	for arr in (i + P, i - P):
		s = np.sort(arr, axis=-1)
		ok &= (np.diff(s, axis=-1) != 0).all(axis=-1)
	return P, ok


def plot(n=11, G=900, path='diag_waves/fig_sturmian.png'):
	P, ok = grid(n, G)
	g = np.linspace(0, 1, G, endpoint=False) + .5 / G
	sols = {p.tobytes() for p in P[ok]}

	fig = plt.figure(figsize=(17, 5.8))

	a = fig.add_subplot(1, 3, 1)
	a.imshow(ok.T, origin='lower', extent=(0, 1, 0, 1), aspect='equal',
			 cmap=plt.cm.colors.ListedColormap([C_BAD, C_GOOD]), interpolation='nearest')
	a.set_xlabel('$\\alpha$'); a.set_ylabel('$\\beta$')
	a.set_title(f'N={n}: el plano de parámetros\n'
				f'naranja = da solución ({ok.mean():.0%} del área)', fontsize=10.5)

	a2 = fig.add_subplot(1, 3, 2)
	marg = ok.mean(axis=1)
	a2.fill_between(g, 0, marg, color=C_GOOD, step='mid')
	for num, den in [(1,5),(1,4),(2,7),(1,3),(3,8),(2,5),(3,7),(1,2)]:
		for v in (num/den, 1-num/den):
			a2.axvline(v, color='#1c7ed6', lw=.8, ls=':')
	a2.text(.5, .93, 'líneas azules: fracciones $p/q$ con $q\\leq 8$',
			transform=a2.transAxes, ha='center', fontsize=8.5, color='#1c7ed6')
	a2.set_xlim(0, 1); a2.set_ylim(0, 1.05)
	a2.set_xlabel('$\\alpha$'); a2.set_ylabel('fracción de $\\beta$ que da solución')
	a2.set_title('Las bandas son casi verticales:\n$\\alpha$ decide, $\\beta$ apenas importa',
				 fontsize=10.5)

	a3 = fig.add_subplot(1, 3, 3)
	picks, seen = [], set()
	for ia in np.linspace(0, G - 1, 220).astype(int):
		for ib in (G // 3, 2 * G // 3, G // 7):
			if not ok[ia, ib]: continue
			k = P[ia, ib].tobytes()
			if k in seen: continue
			seen.add(k); picks.append((P[ia, ib].copy(), g[ia], g[ib]))
			break
		if len(picks) == 4: break
	board = np.zeros((2 * n + 1, 2 * n + 1))
	for j, (p_, al, be) in enumerate(picks):
		r0, c0 = (j // 2) * (n + 1), (j % 2) * (n + 1)
		for r, c in enumerate(p_): board[r0 + r, c0 + c] = 1
	a3.imshow(board, cmap=plt.cm.colors.ListedColormap(['#f8f9fa', C_GOOD]),
			  aspect='equal', interpolation='nearest')
	a3.set_xticks([]); a3.set_yticks([])
	a3.set_title('4 soluciones de bandas distintas\n' +
				 '   '.join(f'$\\alpha$={a_:.3f}' for _, a_, _b in picks), fontsize=9)

	fig.suptitle(f'N={n}: rotación Sturmiana  $x_i=(\\alpha i+\\beta)$ mod 1  ·  '
				 f'{len(sols)} soluciones de {len({p.tobytes() for p in P.reshape(-1,n)})} '
				 f'permutaciones de la familia', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .92))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot(11))
