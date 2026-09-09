"""
Las helices de Costas: una por desfase.

Para cada desfase d,  Z_d(x) = sum_i exp(i*(p[i+d]-p[i])*x)
y  <|Z_d|^2> = (N-d) + 2*conflictos_d.

Costas  <=>  el perfil de energias es EXACTAMENTE la recta N-d.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import permutations

from diag_waves.costas import is_costas

C_GOOD, C_BAD = '#e8590c', '#adb5bd'


def helices(p, pts=None):
	n = len(p); M = pts or 6 * n
	x = 2 * np.pi * np.arange(M) / M          # sin endpoint: la media sale exacta
	return x, [np.exp(1j * np.outer([p[i + d] - p[i] for i in range(n - d)], x)).sum(0)
			   for d in range(1, n)]


def profile(p):
	_, H = helices(p)
	return np.array([(np.abs(z) ** 2).mean() for z in H])


def plot(n=8, path='diag_waves/fig_costas_helix.png'):
	P = list(permutations(range(n)))
	cos = [p for p in P if is_costas(p)]
	no = [p for p in P if not is_costas(p)]
	rng = np.random.default_rng(0)
	ideal = np.array([n - d for d in range(1, n)], float)
	dd = np.arange(1, n)

	fig = plt.figure(figsize=(16, 9.5))

	a = fig.add_subplot(2, 2, 1)
	for p in [no[i] for i in rng.choice(len(no), 300, replace=False)]:
		a.plot(dd, profile(p), color=C_BAD, lw=.4, alpha=.15)
	for p in cos[:80]:
		a.plot(dd, profile(p), color=C_GOOD, lw=1.4, alpha=.6)
	a.plot(dd, ideal, 'k--', lw=2.2, label='la recta $N-d$')
	a.set_xlabel('desfase $d$'); a.set_ylabel('$\\langle |Z_d|^2\\rangle$')
	a.set_title(f'N={n}: el perfil de energia\n'
				f'naranja = los {len(cos)} Costas, todos sobre la recta', fontsize=11)
	a.legend(fontsize=9)

	a2 = fig.add_subplot(2, 2, 2)
	ex_c = np.array([profile(p) - ideal for p in cos[:200]])
	ex_n = np.array([profile(p) - ideal for p in
					 [no[i] for i in rng.choice(len(no), 800, replace=False)]])
	a2.hist(ex_n.sum(1), bins=40, color=C_BAD, label=f'no Costas')
	a2.hist(ex_c.sum(1), bins=[-.5, .5], color=C_GOOD, label=f'{len(cos)} Costas')
	a2.set_yscale('log'); a2.set_xlabel('exceso total sobre la recta')
	a2.set_ylabel('permutaciones (log)')
	a2.set_title('El exceso: Costas = 0 exacto, resto $\\geq$ 2', fontsize=11)
	a2.legend(fontsize=9)

	for j, (p, lab, col) in enumerate(((cos[0], 'COSTAS', C_GOOD),
									   (no[3], 'NO Costas', '#495057'))):
		ax = fig.add_subplot(2, 2, 3 + j, projection='3d')
		x, H = helices(p, pts=900)
		cm = plt.cm.viridis(np.linspace(.1, .9, len(H)))
		for k, z in enumerate(H):
			ax.plot(x, z.real, z.imag, color=cm[k], lw=1.1)
		ax.set_title(f'{lab}  {list(p)}\nlas {len(H)} helices, una por desfase',
					 fontsize=10, color=col)
		ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
		for pane in (ax.xaxis, ax.yaxis, ax.zaxis): pane.pane.set_visible(False)
		ax.view_init(elev=20, azim=-62)

	fig.suptitle('Costas: una hélice por desfase, y su perfil de energía', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .95))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot(8))
