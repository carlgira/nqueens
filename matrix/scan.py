"""Escáner rápido (sin guardar el grafo completo) para N grandes.

Para cada N: soluciones, órbitas D4, y para cada solución a calcula en bloque T = P_b P_a^T con todas las b
y aplica criterios vectorizados. Siempre excluye aristas entre soluciones de la misma órbita D4.

Criterios:  involucion (T^2 = id)   y   afin (T(x) = a*x + b mod N)
Uso: python scan.py 5 6 7 8 9 10 11
"""
import sys, time
from math import gcd
import numpy as np
from build_graph import solve, sym_images, is_toroidal


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]; x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def labels(self):
        roots = np.array([self.find(i) for i in range(len(self.p))])
        _, lab = np.unique(roots, return_inverse=True)
        return lab


def perm_sign(q):
    n = len(q); seen = [False] * n; s = 1
    for i in range(n):
        if not seen[i]:
            l, j = 0, i
            while not seen[j]:
                seen[j] = True; j = q[j]; l += 1
            if l % 2 == 0:
                s = -s
    return s


def scan(n):
    S = solve(n).astype(np.int64); m = len(S)
    index = {tuple(s): i for i, s in enumerate(S)}
    uf_orb = UF(m)
    for i, s in enumerate(S):
        for im in sym_images(s):
            uf_orb.union(i, index[tuple(im)])
    orbit = uf_orb.labels(); n_orb = int(orbit.max()) + 1

    tor = is_toroidal(S)
    ar = np.arange(n)
    rows = np.arange(m)[:, None]
    uf_inv, uf_aff = UF(m), UF(m)
    n_inv = n_aff = 0
    aff_class = {}                      # (a,b) -> lista de aristas
    for a in range(m):
        T = np.empty((m, n), dtype=np.int64)
        T[rows, S[a][None, :]] = S           # T[b, S[a][i]] = S[b][i]
        notsym = (orbit != orbit[a]) & ~(tor[a] & tor)      # sin simetrías y sin toroidal-toroidal
        # involución: T[T[x]] == x
        inv = (np.take_along_axis(T, T, axis=1) == ar).all(axis=1) & notsym
        for b in np.nonzero(inv)[0]:
            uf_inv.union(a, int(b)); n_inv += 1
        # afín mod n: T[x] = (k*x + c) % n
        c = T[:, 0]; k = (T[:, 1] - c) % n
        aff = (T == (k[:, None] * ar[None, :] + c[:, None]) % n).all(axis=1) & notsym
        for b in np.nonzero(aff)[0]:
            uf_aff.union(a, int(b)); n_aff += 1
            aff_class.setdefault((int(k[b]), int(c[b])), []).append((a, int(b)))

    def summarize(uf):
        lab = uf.labels(); sizes = np.bincount(lab)
        return len(sizes), sorted(sizes.tolist(), reverse=True)

    res = dict(n=n, m=m, orbits=n_orb, tor=int(tor.sum()))
    res["inv"] = (n_inv, *summarize(uf_inv), uf_inv.labels())
    res["aff"] = (n_aff, *summarize(uf_aff))
    # mejor (k,c) individual
    best = sorted(aff_class.items(), key=lambda kv: -len(kv[1]))[:3]
    cls = []
    for (k, c), edges in best:
        u = UF(m)
        for x, y in edges:
            u.union(x, y)
        cls.append(((k, c), len(edges), *summarize(u)))
    res["aff_best"] = cls
    res["n_aff_classes"] = len(aff_class)
    res["signs"] = np.array([perm_sign(s) for s in S])
    return res, S


def fmt(sizes):
    return sizes if len(sizes) <= 10 else sizes[:10] + ["..."]


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        t0 = time.time()
        r, S = scan(n)
        e, c, sz, lab = r["inv"]
        print(f"\n=== N={n}: {r['m']} soluciones ({r['tor']} toroidales), {r['orbits']} órbitas D4  ({time.time()-t0:.1f}s)")
        print(f"  involución : aristas={e:>7}  comps={c:>4}  tamaños={fmt(sz)}")
        # ¿las componentes de la involución coinciden con el signo de la permutación?
        signs = r["signs"]
        mix = sum(1 for k in range(c) if len(set(signs[lab == k].tolist())) > 1)
        print(f"             componentes con signos mezclados: {mix}/{c}")
        e, c, sz = r["aff"]
        print(f"  afín mod N : aristas={e:>7}  comps={c:>4}  tamaños={fmt(sz)}  clases (k,c) usadas={r['n_aff_classes']}")
        for (kc, ne, cc, szz) in r["aff_best"]:
            print(f"     T(x)={kc[0]}x+{kc[1]} mod {n}: aristas={ne}, comps={cc}, tamaños={fmt(szz)}")
