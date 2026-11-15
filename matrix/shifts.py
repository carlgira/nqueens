"""Desplazamientos cíclicos como generadores de caminos entre soluciones de N-reinas.

Generadores (siempre sin aristas entre soluciones de la misma órbita D4):
  columnas : q'(i) = (q(i) + c) mod N
  filas    : q'(i) = q((i + r) mod N)
  ambos    : combinación (r, c)
Para cada uno: componentes, tamaños, y para un desplazamiento fijo la estructura de cadenas
(grafo funcional: desde q se pasa a q+c mientras siga siendo solución).

Uso: python shifts.py 5 6 7 8 9 10 11 12
"""
import sys
from collections import Counter
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

    def sizes(self):
        roots = np.array([self.find(i) for i in range(len(self.p))])
        return np.bincount(np.unique(roots, return_inverse=True)[1])


def keys(S, n):
    return (S * (n ** np.arange(n, dtype=np.int64))).sum(axis=1)


def run(n):
    S = solve(n).astype(np.int64); m = len(S)
    K = keys(S, n); order = np.argsort(K); Ks = K[order]

    def lookup(Q):
        kq = keys(Q, n)
        pos = np.searchsorted(Ks, kq); pos[pos == m] = 0
        hit = Ks[pos] == kq
        return np.where(hit, order[pos], -1)

    idx = {tuple(s): i for i, s in enumerate(S)}
    uo = UF(m)
    for i, s in enumerate(S):
        for im in sym_images(s):
            uo.union(i, idx[tuple(im)])
    orbit = np.array([uo.find(i) for i in range(m)])
    tor = is_toroidal(S)

    def edges_for(r, c):
        Q = (S[:, (np.arange(n) + r) % n] + c) % n
        j = lookup(Q)
        a = np.nonzero(j >= 0)[0]
        b = j[a]
        keep = (orbit[a] != orbit[b]) & ~(tor[a] & tor[b])
        return a[keep], b[keep]

    def union_over(pairs):
        u = UF(m); ne = 0
        for r, c in pairs:
            a, b = edges_for(r, c)
            ne += len(a)
            for x, y in zip(a, b):
                u.union(int(x), int(y))
        sz = u.sizes()
        nontriv = sorted([int(s) for s in sz if s > 1], reverse=True)
        return ne, len(sz), nontriv

    cols = [(0, c) for c in range(1, n)]
    rows = [(r, 0) for r in range(1, n)]
    both = [(r, c) for r in range(n) for c in range(n) if (r, c) != (0, 0)]

    print(f"\n=== N={n}: {m} soluciones ({int(tor.sum())} toroidales), {len(set(orbit.tolist()))} órbitas D4")
    for name, pairs in (("columnas", cols), ("filas", rows), ("ambos", both)):
        ne, nc, nt = union_over(pairs)
        covered = sum(nt)
        cnt = Counter(nt)
        top = sorted(cnt.items(), key=lambda kv: -kv[0])[:6]
        print(f"  {name:<9} aristas={ne:>7}  comps={nc:>5}  soluciones en comps no triviales={covered:>5} "
              f"({100*covered/m:4.1f}%)  mayor={nt[0] if nt else 0}  tamaños(tam×cuántas)={top}")

    # cadenas con un desplazamiento fijo de columnas: q -> q+1 -> q+2 ... mientras sea solución
    for c in (1,):
        nxt = np.full(m, -1)
        a, b = edges_for(0, c)
        nxt[a] = b
        has_prev = np.zeros(m, bool); has_prev[b] = True
        starts = [i for i in range(m) if nxt[i] >= 0 and not has_prev[i]]
        lens = []
        for s in starts:
            L, cur = 1, s
            while nxt[cur] >= 0 and L <= n:
                cur = nxt[cur]; L += 1
            lens.append(L)
        cyc = int(((nxt >= 0) & has_prev).sum())   # nodos con entrada y salida (interiores o en ciclo)
        print(f"  cadenas con x+{c}: {len(starts)} cadenas abiertas, longitudes={dict(Counter(lens))}, "
              f"nodos interiores/ciclo={cyc}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(int(a))
