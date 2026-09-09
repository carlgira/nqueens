"""
El cilindro: diagonal -> ángulo (mod 2N-1), solución -> altura.

Si el desplazamiento cíclico fuera una simetría del problema, el dibujo sería
invariante al girarlo. No lo es: el perfil angular tiene una meseta con las
esquinas suprimidas, así que el cilindro tiene un 'ecuador' denso y dos polos
vacíos. Esa asimetría es exactamente la razón de que rotar mate las soluciones.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from diag_waves.diag_waves import channels, shift

C_PLUS, C_MINUS = '#e8590c', '#0ca678'


def sol_spectra(n):
	out, col, dp, dm, cur = set(), [False] * n, set(), set(), []
	def rec(r):
		if r == n:
			a, b = shift(*channels(list(cur)), n)
			out.add((tuple(sorted(a)), tuple(sorted(b)))); return
		for c in range(n):
			if col[c] or (r + c) in dp or (r - c) in dm: continue
			col[c] = True; dp.add(r + c); dm.add(r - c); cur.append(c)
			rec(r + 1); col[c] = False; dp.discard(r + c); dm.discard(r - c); cur.pop()
	rec(0)
	return sorted(out)


def plot(n=9, path='diag_waves/fig_cylinder.png'):
	S = sol_spectra(n)
	m = 2 * n - 1							# diagonales por banda
	ang = lambda k: 2 * np.pi * (k - 1) / m

	fig = plt.figure(figsize=(17, 6))

	# --- el cilindro
	ax = fig.add_subplot(1, 3, 1, projection='3d')
	th = np.linspace(0, 2 * np.pi, 120)
	for z in (0, len(S)):
		ax.plot(np.cos(th), np.sin(th), z, color='#dee2e6', lw=1)
	for j, (A, b) in enumerate(S):
		for k in A:
			a = ang(k); ax.scatter(np.cos(a), np.sin(a), j, color=C_PLUS, s=3, alpha=.5,
								   depthshade=False)
		for k in b:
			a = ang(k); ax.scatter(np.cos(a) * .55, np.sin(a) * .55, j, color=C_MINUS,
								   s=3, alpha=.5, depthshade=False)
	ax.set_title(f'N={n}: {len(S)} espectros apilados\n'
				 f'ángulo = diagonal (mod {m}), altura = solución', fontsize=10.5)
	ax.set_zlabel('solución'); ax.set_xticks([]); ax.set_yticks([])
	ax.view_init(elev=18, azim=-60)

	# --- desenrollado
	a2 = fig.add_subplot(1, 3, 2)
	img = np.zeros((len(S), 2 * m))
	for j, (A, b) in enumerate(S):
		for k in A: img[j, k - 1] = 1
		for k in b: img[j, m + k - 1] = 2
	a2.imshow(img, aspect='auto', cmap=plt.cm.colors.ListedColormap(
		['#f8f9fa', C_PLUS, C_MINUS]), interpolation='nearest')
	a2.axvline(m - .5, color='k', lw=1.2)
	a2.set_xlabel(f'diagonal   ($d^+$ 1..{m}  |  $d^-$ 1..{m})')
	a2.set_ylabel('solución (orden lexicográfico)')
	a2.set_title('El cilindro desenrollado', fontsize=10.5)

	# --- perfil angular: por qué rotar no funciona
	a3 = fig.add_subplot(1, 3, 3, projection='polar')
	prof = img[:, :m].mean(0)
	angs = np.array([ang(k + 1) for k in range(m)])
	a3.plot(np.append(angs, angs[0]), np.append(prof, prof[0]), 'o-',
			color=C_PLUS, ms=4, lw=1.6, label='$\\langle c_k\\rangle$ real')
	a3.plot(np.append(angs, angs[0]), np.full(m + 1, prof.mean()), '--',
			color='#495057', lw=1.4, label=f'uniforme = {prof.mean():.2f}')
	a3.set_title('Perfil angular: NO es uniforme\n'
				 '→ girar el cilindro rompe las soluciones', fontsize=10.5, pad=18)
	a3.legend(fontsize=8.5, loc='lower right', bbox_to_anchor=(1.15, -.08))

	fig.tight_layout()
	fig.savefig(path, dpi=150, bbox_inches='tight')
	print(f'N={n}: {len(S)} espectros · perfil angular '
		  f'min {prof.min():.2f} max {prof.max():.2f} (uniforme {prof.mean():.2f})')
	return path


if __name__ == '__main__':
	print(plot(9))
