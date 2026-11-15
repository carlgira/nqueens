"""Ejemplos de soluciones de N=11 que no salen por ninguna regla -> fig_sin_regla_N11.png
Negro: reina que coincide con la toroidal de un tramo mas cercana; azul: reina fuera de ella;
circulo naranja: donde estaria en esa toroidal."""
import json, random
from reflejos import all_by_rule, tor
from desplazamientos import build, build3
from escaleras import family, family2
from reglas import closure
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
N = 11
K = {tuple(r['solucion']): r for r in json.load(open(f'data/waves_N{N}_nodeg.json'))['soluciones']}
cov = {s for s in K if tor(s)}
for S in (set(family(N)) | set(family2(N)), set(all_by_rule(N)), set(build(N)), set(build3(N, True))):
    cov |= closure(S, N)
lin = [(a, c, tuple((a * i + c) % N for i in range(N))) for a in range(2, N - 1) for c in range(N)]
sg = lambda x: (x % N) if (x % N) <= N // 2 else (x % N) - N


def info(s):
    al = [sg(s[i + 1] - s[i]) for i in range(N - 1)]
    d, a, c, t = min((sum(x != y for x, y in zip(s, t)), a, c, t) for a, c, t in lin)
    return al, d, a, t


un = [s for s in K if s not in cov]
rng = random.Random(4)
D = lambda s: len(set(info(s)[0]))


def pick(cond, n):
    L = [s for s in un if cond(s)]
    rng.shuffle(L)
    return L[:n]


ex = [('3 alfas (escalones no opuestos)', s) for s in pick(lambda s: D(s) == 3, 2)]
ex += [('a 3 reinas de una toroidal', s) for s in pick(lambda s: info(s)[1] == 3, 2)]
ex += [('4 alfas', s) for s in pick(lambda s: D(s) == 4 and info(s)[1] > 3, 1)]
ex += [('típica de 5 ondas', s) for s in pick(lambda s: K[s]['ondas_min'] == 5 and D(s) >= 5, 2)]
ex += [('6 ondas, lejos de toda toroidal', s) for s in pick(lambda s: K[s]['ondas_min'] == 6 and info(s)[1] >= 7, 1)]
fig, axs = plt.subplots(2, 4, figsize=(15, 9.4))
for ax, (lab, s) in zip(axs.ravel(), ex):
    al, d, a, t = info(s)
    for i in range(N + 1):
        ax.plot([-.5, N - .5], [i - .5, i - .5], color='#e4e3df', lw=.6)
        ax.plot([i - .5, i - .5], [-.5, N - .5], color='#e4e3df', lw=.6)
    ax.scatter(t, range(N), s=120, facecolors='none', edgecolors='#eb6834', lw=1.3, zorder=2)
    same = [i for i in range(N) if s[i] == t[i]]
    dif = [i for i in range(N) if s[i] != t[i]]
    ax.scatter([s[i] for i in same], same, s=60, c='#0b0b0b', zorder=3)
    ax.scatter([s[i] for i in dif], dif, s=60, c='#2a78d6', zorder=3)
    ax.plot(s, range(N), color='#52514e', lw=.7, alpha=.5, zorder=1)
    ax.set_xlim(-.5, N - .5); ax.set_ylim(N - .5, -.5); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{lab}\n{len(set(al))} alfas · {K[s]['ondas_min']} ondas · a {d} reinas de una toroidal", fontsize=9.5, loc='left')
    ax.set_xlabel('alfas: ' + ' '.join(f'{x:+d}' for x in al), fontsize=8, color='#52514e')
fig.subplots_adjust(hspace=0.28, wspace=0.08, top=0.93, bottom=0.05, left=0.02, right=0.98)
fig.savefig('fig_sin_regla_N11.png', dpi=120, facecolor='white')
