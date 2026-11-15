import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensor import valid, solutions
from moran import tower, window, moran_points

lin = lambda m, c: np.array([(c * j) % m for j in range(m)])
taus = [solutions(8)[0], lin(5, 2), lin(7, 3), lin(11, 3), lin(13, 5)]
names = ["sol. de N=8\n(no toroidal)", "2j mod 5", "3j mod 7", "3j mod 11", "5j mod 13"]
sizes = [len(t) for t in taus]
P = tower(taus)
assert valid(P)
print(f"torre N={len(P)} = {' x '.join(map(str,sizes))}  valida={valid(P)}")

fig, ax = plt.subplots(2, 5, figsize=(19, 8.6), facecolor='#08080c',
                       gridspec_kw={'height_ratios': [1.6, 1]})
for c, (k, nm) in enumerate(zip(range(1, 6), names)):
	x, y, w = window(P, sizes, k)
	m = sizes[k - 1]; b = w // m
	a = ax[0, c]
	a.scatter(x, y, s=max(0.6, 2000 / w), c='#ffb347', marker='s', linewidths=0)
	for g in range(1, m):
		a.axhline(g * b, color='#3a3a4a', lw=0.5); a.axvline(g * b, color='#3a3a4a', lw=0.5)
	a.set_xlim(0, w); a.set_ylim(0, w); a.set_aspect('equal')
	a.set_title(f"nivel {k}   ventana {w}x{w}", color='#e8e8f0', fontsize=10)
	a.set_xticks([]); a.set_yticks([]); a.set_facecolor('#08080c')

	# coarse-grain: que bloque m x m esta ocupado -> tiene que ser tau_k
	blocks = np.zeros((m, m), np.uint8)
	blocks[y // b, x // b] = 1
	got = blocks.argmax(0)
	assert np.array_equal(got, taus[k - 1]), (k, got, taus[k - 1])
	a = ax[1, c]
	a.imshow(blocks, origin='lower', cmap='magma', vmin=0, vmax=1.4, interpolation='nearest')
	a.set_title(f"{nm}   ({m}x{m})", color='#00ff9c', fontsize=10)
	a.set_xticks([]); a.set_yticks([]); a.set_facecolor('#08080c')

fig.suptitle(f"UNA solucion de N={len(P)}: arriba el zoom, abajo la solucion que gobierna esa escala",
             color='#e8e8f0', fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig('fig_moran_zoom.png', dpi=130, facecolor='#08080c')
print('fig_moran_zoom.png  (coarse-grain verificado == tau_k en los 5 niveles)')

seq = [lin(7, 3), lin(5, 2), lin(11, 3), lin(13, 5), lin(7, 2), lin(5, 3)]
seq_names = ["3j mod 7", "2j mod 5", "3j mod 11", "5j mod 13", "2j mod 7", "3j mod 5"]
fig, ax = plt.subplots(1, 4, figsize=(20, 5.6), facecolor='#08080c')
for a, K in zip(ax, (3, 4, 5, 6)):
	z = moran_points(seq[:K], dim=1.85)
	c = np.repeat(np.arange(len(seq[0])), len(z) // len(seq[0]))
	a.scatter(z.real, z.imag, s=max(0.05, 700 / len(z) ** 0.5), c=c, cmap='turbo', linewidths=0)
	a.set_aspect('equal'); a.set_xticks([]); a.set_yticks([]); a.set_facecolor('#08080c')
	a.set_title(f"{K} niveles - {len(z)} puntos\n" + " > ".join(seq_names[:K]),
	            color='#e8e8f0', fontsize=8)
fig.suptitle("el mismo apilado en version espiral: un nivel de IFS por solucion",
             color='#e8e8f0', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('fig_moran_spiral.png', dpi=130, facecolor='#08080c')
print('fig_moran_spiral.png')
