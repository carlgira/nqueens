"""
El analogo en 3D: emisores sobre una esfera.

En 1D las rendijas estan en posiciones enteras y el testigo es exacto:

	<I> = N + 2*(pares en la MISMA posicion) = N + 2*conflictos      >= N

En 3D con N emisores coherentes en r_1..r_N, promediando |W(k)|^2 sobre todas
las direcciones de emision se obtiene la potencia total radiada, tambien en
forma cerrada:

	P = N + 2*sum_{i<j} sinc(k*d_ij),      d_ij = |r_i - r_j|

Es el mismo testigo con el nucleo cambiado: la DELTA de 1D se convierte en un
SINC. Y eso cambia el problema por completo.

	sinc(0) = 1        dos emisores encima: superradiancia, P sube (como un
					   conflicto de reinas)
	sinc(x) = 0        en x = pi, 2pi, ...: el par NO contribuye. Es el
					   equivalente exacto de "no hay conflicto"
	sinc(x) < 0        en (pi, 2pi) y demas lobulos: el par RESTA

Esa tercera opcion no existe en 1D. En N-reinas lo mejor posible es P = N; en
3D se puede bajar de N -- es la SUBRRADIANCIA de Dicke. El sistema tiene un
problema de equilibrio de verdad, con un potencial de pares oscilante y con
signo, no una restriccion combinatoria.

Y las "franjas" en 3D no son lineas: son la funcion I(k) sobre la esfera de
direcciones, con casquetes brillantes y oscuros. El numero de canales
independientes que se pueden controlar es ~(kR)^2 (armonicos esfericos hasta
grado L ~ kR), asi que con N emisores solo se pueden apagar ~N direcciones
efectivas.
"""

import numpy as np

GREY, BLUE, ORANGE, PURPLE, GREEN = '#dee2e6', '#1c7ed6', '#e8590c', '#7048e8', '#0ca678'


def sinc(x):
	return np.sinc(x / np.pi)					# np.sinc(t) = sin(pi t)/(pi t)


def power(R, k):
	"""P = N + 2*sum_{i<j} sinc(k d_ij). Exacto, sin integrar."""
	D = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
	S = sinc(k * D)
	return float(S.sum())						# la diagonal aporta N (sinc(0)=1)


def far_field(R, k, ndir=200):
	"""I(dir) sobre la esfera de direcciones, en malla lat-lon."""
	th = np.linspace(0, np.pi, ndir)
	ph = np.linspace(0, 2 * np.pi, 2 * ndir)
	T, P = np.meshgrid(th, ph, indexing='ij')
	K = k * np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
	W = np.exp(1j * (K @ R.T)).sum(-1)
	return T, P, np.abs(W) ** 2


def relax(n, k, radius=1.0, steps=4000, lr=.02, seed=0, on_sphere=True):
	"""Bajar P moviendo los emisores. Si on_sphere, quedan en la superficie."""
	rng = np.random.default_rng(seed)
	R = rng.normal(size=(n, 3))
	R /= np.linalg.norm(R, axis=1, keepdims=True)
	R *= radius
	hist = [power(R, k)]
	for _ in range(steps):
		D = R[:, None, :] - R[None, :, :]
		d = np.linalg.norm(D, axis=-1)
		np.fill_diagonal(d, 1.0)				# valor seguro; D[i,i]=0 anula el termino
		x = k * d
		# d/dx sinc(x) = (x cos x - sin x)/x^2
		g = (x * np.cos(x) - np.sin(x)) / x ** 2
		G = (2 * g * k / d)[:, :, None] * D			# gradiente de P respecto a r_i
		G = G.sum(1)
		if on_sphere:
			u = R / np.linalg.norm(R, axis=1, keepdims=True)
			G -= (G * u).sum(1, keepdims=True) * u	# proyectar al plano tangente
		R = R - lr * G
		if on_sphere:
			R = radius * R / np.linalg.norm(R, axis=1, keepdims=True)
		hist.append(power(R, k))
	return R, np.array(hist)


def report(n=12):
	print(f'N = {n} emisores.  P = N + 2*sum sinc(k d).  En 1D el minimo seria P = N = {n}.')
	print(f'{"kR":>6} {"P inicial":>11} {"P relajado":>11} {"P/N":>8}  {"regimen"}')
	for kR in (0.5, 1, 2, 4, 8, 16, 32):
		R, h = relax(n, kR, radius=1.0)
		reg = ('superradiante' if h[-1] > n * 1.5 else
			   'SUBRRADIANTE' if h[-1] < n * .5 else 'intermedio')
		print(f'{kR:>6} {h[0]:>11.3f} {h[-1]:>11.3f} {h[-1]/n:>8.3f}  {reg}')
	print('\nEn 1D (N-reinas) P/N >= 1 siempre: el nucleo delta no puede ser negativo.')
	print('En 3D el sinc si, y por eso hay equilibrio de verdad en vez de restriccion.')


def _mpl():
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	return plt


def plot(n=12, path='diag_waves/fig_sphere.png'):
	plt = _mpl()
	fig = plt.figure(figsize=(15.5, 8.4))

	# --- el nucleo: delta contra sinc
	ax = fig.add_subplot(2, 3, 1)
	x = np.linspace(0, 25, 800)
	ax.plot(x, sinc(x), color=GREEN, lw=1.8)
	ax.axhline(0, color='k', lw=.8)
	ax.plot([0], [1], 'o', color=ORANGE, ms=9)
	ax.annotate('1D: solo existe\neste punto (delta)', (0, 1),
				textcoords='offset points', xytext=(24, -6), fontsize=8, color=ORANGE)
	ax.fill_between(x, sinc(x), 0, where=sinc(x) < 0, color=BLUE, alpha=.25)
	ax.annotate('3D: aqui el par RESTA\n(subrradiancia)', (7.7, -.12),
				textcoords='offset points', xytext=(20, -34), fontsize=8, color=BLUE)
	ax.set_xlabel('k·d'); ax.set_ylabel('aporte del par')
	ax.set_title('el nucleo del testigo\n1D: delta   3D: sinc', fontsize=10)

	# --- relajacion
	ax = fig.add_subplot(2, 3, 2)
	for kR, col in ((1, ORANGE), (4, PURPLE), (16, GREEN)):
		R, h = relax(n, kR)
		ax.plot(h / n, color=col, lw=1.8, label=f'kR = {kR}')
	ax.axhline(1, color='k', ls='--', lw=1.2)
	ax.annotate('P = N: el minimo de 1D', (len(h) * .45, 1.05), fontsize=8)
	ax.set_yscale('log'); ax.set_xlabel('paso'); ax.set_ylabel('P / N')
	ax.legend(fontsize=8); ax.set_title(f'bajando la potencia con {n} emisores\n'
										'en 3D se puede cruzar la linea', fontsize=10)

	# --- configuraciones y patrones
	for c, kR in enumerate((1, 4, 16)):
		R, h = relax(n, kR)
		a = fig.add_subplot(2, 3, 4 + c, projection='3d')
		u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
		a.plot_surface(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
					   color=GREY, alpha=.18, linewidth=0)
		a.scatter(R[:, 0], R[:, 1], R[:, 2], s=70, color=ORANGE, depthshade=False)
		a.set_title(f'kR = {kR}   P/N = {h[-1]/n:.3f}', fontsize=10)
		a.set_xticks([]); a.set_yticks([]); a.set_zticks([])
		a.set_box_aspect((1, 1, 1))

	# --- las franjas en 3D: I sobre la esfera de direcciones
	ax = fig.add_subplot(2, 3, 3)
	R, h = relax(n, 8)
	T, P, I = far_field(R, 8)
	im = ax.pcolormesh(P, np.cos(T), I / I.max(), cmap='inferno', shading='auto')
	plt.colorbar(im, ax=ax, label='I / I max')
	ax.set_xlabel('azimut'); ax.set_ylabel('cos(polar)')
	ax.set_title('las "franjas" en 3D: casquetes sobre la\nesfera de direcciones (kR=8)',
				 fontsize=10)

	fig.suptitle('emisores sobre una esfera: el mismo testigo con el nucleo cambiado', y=.99)
	fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
	print(f'-> {path}')


if __name__ == '__main__':
	report()
	plot()
