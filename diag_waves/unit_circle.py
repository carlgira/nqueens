"""Radio 1 <=> solución. Y las funciones X(x), Y(x) por separado."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.diag_waves import band_series, channels, conflicts

C_GOOD, C_BAD = '#e8590c', '#495057'


def plot(n=9, good=(0,2,5,7,1,3,8,6,4), bad=(0,1,3,6,2,8,4,7,5),
		 xs=1.0, path='diag_waves/fig_unit_circle.png'):
	M = 4096
	x = np.linspace(0, 2 * np.pi, M, endpoint=False)
	fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.8))

	for a, p, name, col in ((ax[0], list(good), 'SOLUCIÓN', C_GOOD),
							(ax[1], list(bad), 'NO SOLUCIÓN', C_BAD)):
		c = band_series(p)
		ks = np.nonzero(c)[0]
		dp, dm = channels(p)
		nc = conflicts(dp) + conflicts(dm)
		rmax = int(c.max())
		th = np.linspace(0, 2 * np.pi, 300)
		for r in range(1, rmax + 1):
			a.plot(r * np.cos(th), r * np.sin(th), color='#ced4da', lw=1,
				   ls='-' if r == 1 else '--')
			a.annotate(f'r={r}', (0, r), fontsize=8, color='#868e96',
					   ha='center', va='bottom')
		cm = plt.cm.turbo(np.linspace(0, 1, len(c)))
		for k in ks:
			v = c[k] * np.exp(1j * k * xs)
			a.plot([0, v.real], [0, v.imag], color=cm[k], lw=1.5, alpha=.7)
			a.scatter(v.real, v.imag, color=cm[k], s=150, zorder=3,
					  edgecolor='k', linewidth=.6)
			a.annotate(str(k), (v.real, v.imag), fontsize=6.5, ha='center',
					   va='center', color='w', fontweight='bold', zorder=4)
		a.set_aspect('equal'); a.set_xlabel('Re'); a.set_ylabel('Im')
		a.set_title(f'{name}  {p}\n{len(ks)} términos · radios {sorted(set(c[ks].astype(int)))}'
					f' · conflictos = {nc}', fontsize=10.5, color=col)

	# --- X(x) y Y(x)
	a = ax[2]
	K = np.arange(4 * n)
	for p, name, col, ls in ((list(good), 'sol', C_GOOD, '-'),
							 (list(bad), 'no sol', C_BAD, '-')):
		c = band_series(p)
		X = c @ np.cos(np.outer(K, x))
		Y = c @ np.sin(np.outer(K, x))
		dp, dm = channels(p); nc = conflicts(dp) + conflicts(dm)
		a.plot(x, X, color=col, lw=1.4, ls=ls,
			   label=f'X ({name}):  $\\langle X^2\\rangle$ = {(X**2).mean():.0f} = N+{nc}')
		a.plot(x, Y, color=col, lw=1.1, ls=':',
			   label=f'Y ({name}):  $\\langle Y^2\\rangle$ = {(Y**2).mean():.0f} = N+{nc}')
	a.set_xlim(0, 2 * np.pi); a.set_xlabel('x'); a.set_ylabel('amplitud')
	a.set_title('$X(x)=\\sum c_k\\cos kx$   ·   $Y(x)=\\sum c_k\\sin kx$\n'
				'las dos dan el MISMO testigo', fontsize=10.5)
	a.legend(fontsize=8.5, loc='lower center')

	fig.suptitle(f'N={n}: todos los términos en el círculo unidad $\\Leftrightarrow$ es solución',
				 fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .93))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot())
