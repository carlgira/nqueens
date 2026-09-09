"""
La versión discreta y finita:   Z(x_j), j = 0..M-1   ->   c_k

Muestreando la hélice en M puntos equiespaciados x_j = 2*pi*j/M se recupera el
vector de ocupación EXACTO, sin integrales:

	c_k = (1/M) * sum_j  Z(x_j) * exp(-i*k*x_j)

Cuántas muestras hacen falta:

	desde Z (compleja, la hélice)    M >= 4N       (basta M > k_max = 4N-1)
	desde X o Y (la sombra real)     M >= 8N       (real: hace falta M > 2*k_max)

La hélice necesita LA MITAD de muestras que su sombra. Es la primera ventaja
cuantificada del objeto complejo sobre el real.
"""

import numpy as np

from diag_waves.diag_waves import band_series


def sample(perm, m=None):
	"""Las M muestras Z(2*pi*j/M). Devuelve (x_j, Z_j)."""
	c = band_series(perm)
	m = m or 4 * len(perm)
	xj = 2 * np.pi * np.arange(m) / m
	return xj, c @ np.exp(1j * np.outer(np.arange(len(c)), xj))


def recover(z, n):
	"""c_k a partir de las muestras. Exacto si M >= 4N."""
	m = len(z)
	xj = 2 * np.pi * np.arange(m) / m
	return np.rint([(z * np.exp(-1j * k * xj)).sum().real / m
					for k in range(4 * n)]).astype(int)


def recover_from_shadow(y, n, which='sin'):
	"""c_k desde solo X o solo Y. Necesita M >= 8N."""
	m = len(y)
	xj = 2 * np.pi * np.arange(m) / m
	f = np.sin if which == 'sin' else np.cos
	return np.rint([2 * (y * f(k * xj)).sum() / m for k in range(4 * n)]).astype(int)


def min_samples(n):
	return {'helice': 4 * n, 'sombra': 8 * n}


if __name__ == '__main__':
	n = 9
	p = [0, 2, 5, 7, 1, 3, 8, 6, 4]
	c = band_series(p).astype(int)
	for m in (24, 36, 71, 72):
		_, z = sample(p, m)
		ok = np.array_equal(recover(z, n), c)
		oky = np.array_equal(recover_from_shadow(z.imag, n), c)
		print(f'M={m:>3}: desde Z {"EXACTO" if ok else "falla ":>7}   '
			  f'desde Y {"EXACTO" if oky else "falla"}')
	print(f'\nminimos para N={n}: {min_samples(n)}')
	print(f'diagonales ocupadas: {np.nonzero(c)[0].tolist()}')


def plot(n=9, perm=(0, 2, 5, 7, 1, 3, 8, 6, 4), path='diag_waves/fig_sampling.png'):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	perm = list(perm)
	c = band_series(perm).astype(int)
	H = 4 * n
	fig = plt.figure(figsize=(17, 5.8))

	# --- la helice con las M muestras
	ax = fig.add_subplot(1, 3, 1, projection='3d')
	xf = np.linspace(0, 2 * np.pi, 2000)
	Zf = band_series(perm) @ np.exp(1j * np.outer(np.arange(H), xf))
	ax.plot(xf, Zf.real, Zf.imag, color='#adb5bd', lw=.8)
	xj, zj = sample(perm, H)
	ax.scatter(xj, zj.real, zj.imag, color='#e8590c', s=28, depthshade=False)
	ax.set_xlabel('x'); ax.set_ylabel('Re Z'); ax.set_zlabel('Im Z')
	ax.set_title(f'La hélice y sus M = 4N = {H} muestras\n'
				 '$x_j = 2\\pi j/M$', fontsize=11)

	# --- coeficientes recuperados
	a = fig.add_subplot(1, 3, 2)
	k = np.arange(H)
	bad_m = 24
	_, zb = sample(perm, bad_m)
	xb = 2 * np.pi * np.arange(bad_m) / bad_m
	rec_bad = np.array([(zb * np.exp(-1j * kk * xb)).sum().real / bad_m for kk in k])
	a.stem(k, c, linefmt='#adb5bd', markerfmt='o', basefmt=' ', label='$c_k$ real')
	a.plot(k, recover(sample(perm, H)[1], n), 'x', color='#e8590c', ms=11, mew=2.2,
		   label=f'recuperado, M={H}  (exacto)')
	a.plot(k, rec_bad, '.', color='#1c7ed6', ms=9,
		   label=f'recuperado, M={bad_m}  (aliasing)')
	a.axvline(2 * n - .5, color='#ced4da', ls='--', lw=1)
	a.text(n, 1.25, 'banda $d^+$', ha='center', fontsize=9, color='#868e96')
	a.text(3 * n, 1.25, 'banda $d^-$', ha='center', fontsize=9, color='#868e96')
	a.set_xlabel('armónico $k$'); a.set_ylabel('$c_k$')
	a.set_ylim(-.4, 1.5)
	a.set_title('Los coeficientes salen exactos con M = 4N', fontsize=11)
	a.legend(fontsize=8.5, loc='lower right')

	# --- umbral: error vs M
	a = fig.add_subplot(1, 3, 3)
	Ms = np.arange(10, 100)
	eZ, eY = [], []
	for m in Ms:
		xm = 2 * np.pi * np.arange(m) / m
		z = band_series(perm) @ np.exp(1j * np.outer(np.arange(H), xm))
		eZ.append(np.abs([(z * np.exp(-1j * kk * xm)).sum().real / m for kk in k] - c).max())
		eY.append(np.abs([2 * (z.imag * np.sin(kk * xm)).sum() / m for kk in k] - c).max())
	a.semilogy(Ms, np.maximum(eZ, 1e-16), 'o-', color='#e8590c', ms=3,
			   label='desde $Z$ (hélice)')
	a.semilogy(Ms, np.maximum(eY, 1e-16), 's-', color='#1c7ed6', ms=3,
			   label='desde $Y$ (sombra)')
	for m, col, lab in ((4 * n, '#e8590c', f'4N={4*n}'), (8 * n, '#1c7ed6', f'8N={8*n}')):
		a.axvline(m, color=col, ls='--', lw=1.4)
		a.text(m + 1, 1e-6, lab, color=col, fontsize=9, fontweight='bold')
	a.set_xlabel('nº de muestras M'); a.set_ylabel('error máx en $c_k$')
	a.set_title('El umbral: la hélice cuesta la mitad', fontsize=11)
	a.legend(fontsize=9); a.grid(alpha=.25)

	fig.suptitle(f'N={n}, solución {perm}:  $Z(x_j) \\rightarrow c_k$', fontsize=13)
	fig.tight_layout(rect=(0, 0, 1, .93))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path
