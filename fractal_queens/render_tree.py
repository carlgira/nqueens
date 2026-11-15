import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tree import build, profile


def radial_layout(depth, parent, weight):
	"""Reparte el angulo de cada nodo entre sus hijos, proporcional al peso.
	parent[i] < i siempre (orden DFS), asi que basta una pasada en cada sentido."""
	n = len(depth)
	sub = weight.astype(float).copy()
	for i in range(n - 1, 0, -1):
		sub[parent[i]] += sub[i]
	lo = np.zeros(n); hi = np.full(n, 2 * np.pi)
	cur = lo.copy()
	for i in range(1, n):
		p = parent[i]
		span = (hi[p] - lo[p]) * sub[i] / sub[p]
		if cur[p] == 0.0 and lo[p] != 0.0:
			cur[p] = lo[p]
		lo[i] = max(cur[p], lo[p]); hi[i] = lo[i] + span
		cur[p] = hi[i]
	return (lo + hi) / 2


def curved(x0, y0, x1, y1, r0, t0, r1, t1, k=7):
	f = np.linspace(0, 1, k)[None, :]
	r = r0[:, None] * (1 - f) + r1[:, None] * f
	t = t0[:, None] * (1 - f) + t1[:, None] * f
	return np.stack([r * np.cos(t), r * np.sin(t)], -1)


N = 8
d, _, par, al = build(N)
print(f"N={N}: {len(d)} nodos, {(d[al]==N).sum()} soluciones, {100*(1-al.mean()):.0f}% poda")
th = radial_layout(d, par, np.ones(len(d)))
r = 1.0 + d.astype(float)

segs = curved(None, None, None, None, r[par[1:]], th[par[1:]], r[1:], th[1:])
liv = al[1:]

fig = plt.figure(figsize=(13, 13), facecolor='#07070b')
a = fig.add_axes([0.01, 0.03, 0.98, 0.94]); a.set_facecolor('#07070b')
a.add_collection(LineCollection(segs[~liv], colors='#223a5c', linewidths=0.5, zorder=1))
a.add_collection(LineCollection(segs[liv], colors=plt.cm.plasma((d[1:][liv] - 1) / (N - 1)),
                                linewidths=1.3, zorder=2))
leaf = al & (d == N)
a.scatter(r[leaf] * np.cos(th[leaf]), r[leaf] * np.sin(th[leaf]),
          s=26, c='#fff6b0', zorder=3, linewidths=0)
a.set_xlim(-N - 1.5, N + 1.5); a.set_ylim(-N - 1.5, N + 1.5); a.set_aspect('equal')
a.set_xticks([]); a.set_yticks([])
a.text(0, -N - 1.15, f"N={N}   {len(d)} nodos   {(d[al]==N).sum()} soluciones (puntos)   "
                     f"{100*(1-al.mean()):.0f}% del arbol muere antes de llegar",
       color='#8a8a99', ha='center', fontsize=12)
fig.savefig('fig_tree_radial.png', dpi=165, facecolor='#07070b')
print('fig_tree_radial.png')

# N=10 para ver el coral denso
N2 = 10
d2, _, par2, al2 = build(N2)
th2 = radial_layout(d2, par2, np.ones(len(d2)))
r2 = 1.0 + d2.astype(float)
segs2 = curved(None, None, None, None, r2[par2[1:]], th2[par2[1:]], r2[1:], th2[1:])
liv2 = al2[1:]
fig = plt.figure(figsize=(13, 13), facecolor='#07070b')
a = fig.add_axes([0.01, 0.03, 0.98, 0.94]); a.set_facecolor('#07070b')
a.add_collection(LineCollection(segs2[~liv2], colors='#1b2c47', linewidths=0.18, zorder=1))
a.add_collection(LineCollection(segs2[liv2], colors=plt.cm.plasma((d2[1:][liv2] - 1) / (N2 - 1)),
                                linewidths=0.32, zorder=2))
lf = al2 & (d2 == N2)
a.scatter(r2[lf] * np.cos(th2[lf]), r2[lf] * np.sin(th2[lf]), s=3.5, c='#fff6b0', zorder=3, linewidths=0)
a.set_xlim(-N2 - 1.5, N2 + 1.5); a.set_ylim(-N2 - 1.5, N2 + 1.5); a.set_aspect('equal')
a.set_xticks([]); a.set_yticks([])
a.text(0, -N2 - 1.1, f"N={N2}   {len(d2)} nodos   {(d2[al2]==N2).sum()} soluciones   "
                     f"{100*(1-al2.mean()):.0f}% poda", color='#8a8a99', ha='center', fontsize=12)
fig.savefig('fig_tree_radial_n10.png', dpi=165, facecolor='#07070b')
print('fig_tree_radial_n10.png')

# perfil y dimension
Q = {6:4,7:40,8:92,9:352,10:724,11:2680,12:14200}
fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), facecolor='#0d0d12')
rows = []
for n in range(6, 13):
	W, L, tot = profile(n)
	ax[0].semilogy(range(n + 1), W, 'o-', ms=3, lw=1.2, label=f"N={n}")
	rows.append((n, tot, Q[n], np.log(Q[n]) / (n * np.log(n)), 1 - 1.944 / np.log(n)))
ax[0].set_xlabel('profundidad d (reinas colocadas)', color='#8a8a99')
ax[0].set_ylabel('nodos sin conflicto W_d', color='#8a8a99')
ax[0].set_title('perfil del arbol: crece, culmina, colapsa', color='#e8e8f0')
ax[0].legend(fontsize=7, ncol=2, facecolor='#0d0d12', labelcolor='#c8c8d4')
ns = [x[0] for x in rows]
ax[1].plot(ns, [x[3] for x in rows], 'o-', c='#00ff9c', label='log Q(N)/(N log N)  (real)')
ax[1].plot(ns, [x[4] for x in rows], 's--', c='#ff8c42', label='1 - 1.944/ln N  (asintota, Simkin)')
ax[1].set_xlabel('N', color='#8a8a99'); ax[1].set_ylabel('dimension de caja', color='#8a8a99')
ax[1].set_title('dimension del Cantor de soluciones', color='#e8e8f0')
ax[1].legend(fontsize=8, facecolor='#0d0d12', labelcolor='#c8c8d4')
for a_ in ax:
	a_.tick_params(colors='#8a8a99'); a_.set_facecolor('#12121a')
fig.tight_layout(); fig.savefig('fig_tree_profile.png', dpi=135, facecolor='#0d0d12')
print('fig_tree_profile.png')
