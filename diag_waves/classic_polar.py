"""¿Se parecen las curvas polares clásicas a las flores de las soluciones?"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.cylinder import sol_spectra

C_CLS, C_SOL = '#1c7ed6', '#e8590c'


def plot(n=9, path='diag_waves/fig_classic_polar.png'):
	M = 2 * n - 1
	th = 2 * np.pi * np.arange(M) / M
	thc = np.linspace(0, 2 * np.pi, 800)
	S = [A for A, _ in sol_spectra(n)]

	classic = [
		('cardioide\n$r=1+\\cos\\theta$', 1 + np.cos(thc), 1 + np.cos(th)),
		('limaçon\n$r=0.6+\\cos\\theta$', .6 + np.cos(thc), .6 + np.cos(th)),
		('rosa $m=5$\n$r=|\\cos 5\\theta|$', np.abs(np.cos(5*thc)), np.abs(np.cos(5*th))),
		('cónica $e=0.7$\n$r=1/(1+e\\cos\\theta)$', 1/(1+.7*np.cos(thc)), 1/(1+.7*np.cos(th))),
	]

	fig, ax = plt.subplots(2, 4, figsize=(17, 9))
	for j, (name, rc, rd) in enumerate(classic):
		a = ax[0, j]; a.remove()
		p = fig.add_subplot(2, 4, j + 1, projection='polar')
		p.plot(thc, rc, color=C_CLS, lw=2)
		sel = np.argsort(rd)[-n:]
		mask = np.zeros(M, bool); mask[sel] = True
		p.scatter(th[mask], rd[mask], color=C_CLS, s=70, zorder=4, edgecolor='k', lw=.6)
		p.scatter(th[~mask], rd[~mask], color='#ced4da', s=35, zorder=4)
		got = sorted(int(v) + 1 for v in sel)
		p.set_title(f'{name}\nelige {got}', fontsize=7.5, pad=12)
		p.set_yticklabels([]); p.set_xticklabels([])

	for j, A in enumerate(S[::47][:4]):
		a = ax[1, j]; a.remove()
		p = fig.add_subplot(2, 4, 5 + j, projection='polar')
		v = np.zeros(M); v[np.array(A) - 1] = 1
		V = np.fft.fft(v); fr = np.fft.fftfreq(M, 1/M)
		rc = (V[None] * np.exp(1j * np.outer(thc, fr))).sum(1).real / M
		p.plot(thc, rc, color=C_SOL, lw=2)
		m = v > 0
		p.scatter(th[m], v[m], color=C_SOL, s=70, zorder=4, edgecolor='k', lw=.6)
		p.scatter(th[~m], v[~m], color='#ced4da', s=35, zorder=4)
		p.set_title(f'solución real\n{[int(v) for v in A]}', fontsize=7.5, pad=12)
		p.set_yticklabels([]); p.set_xticklabels([])

	fig.text(.5, .96, 'ARRIBA: curvas polares clásicas  ·  '
			 'ABAJO: las flores de soluciones reales',
			 ha='center', fontsize=14)
	fig.text(.5, .5, 'las clásicas eligen bloques contiguos  ·  '
			 'las soluciones eligen conjuntos dispersos',
			 ha='center', fontsize=11, style='italic', color='#495057')
	fig.tight_layout(rect=(0, 0, 1, .94))
	fig.savefig(path, dpi=150, bbox_inches='tight')
	return path


if __name__ == '__main__':
	print(plot())
