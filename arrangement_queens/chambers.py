"""Geometria de la camara que contiene cada solucion.

Disposicion A_N en R^N:  x_i - x_j = c,  c in {0, +(j-i), -(j-i)}  (i<j).
Todo es invariante por traslacion x -> x + t(1,..,1): trabajamos modulo ella
(fijando x_0 = 0), la distancia a la pared x_i-x_j=c es |x_i-x_j-c|/sqrt(2).

Por solucion sigma calculamos:
  bounded  : la camara es un politopo acotado (modulo traslacion).
             Criterio combinatorio: grafo de "pares cercanos" |s_i-s_j|<|i-j|
             conexo  <=>  acotada.  (Se comprueba contra LP.)
  facets   : cuantas de las 3*C(N,2) paredes tocan de verdad la camara.
  inradius : radio de la mayor bola dentro de la camara (Chebyshev).
  depth    : distancia de la propia solucion a la pared mas cercana.
"""
import json, sys
from itertools import combinations, permutations
import numpy as np
from scipy.optimize import linprog

def solutions(N):
    out, cols = [], [0] * N
    def rec(r, used, d1, d2):
        if r == N: out.append(tuple(cols)); return
        for c in range(N):
            if not (used >> c) & 1 and not (d1 >> (r + c)) & 1 and not (d2 >> (r - c + N)) & 1:
                cols[r] = c; rec(r + 1, used | 1 << c, d1 | 1 << (r + c), d2 | 1 << (r - c + N))
    rec(0, 0, 0, 0); return out

def hyperplanes(N):
    return [(i, j, c) for i, j in combinations(range(N), 2) for c in (0, j - i, -(j - i))]

def halfspaces(N, s):
    """filas (a, b): a.x >= b  en coords reducidas x_1..x_{N-1} (x_0=0), una por pared."""
    H = hyperplanes(N); rows = []
    for i, j, c in H:
        sg = np.sign(s[i] - s[j] - c)
        a = np.zeros(N); a[i] += sg; a[j] -= sg
        rows.append((a[1:], sg * c))
    A = np.array([r[0] for r in rows]); b = np.array([r[1] for r in rows])
    return H, A, b

def near_graph_connected(N, s):
    adj = {i: set() for i in range(N)}
    for i, j in combinations(range(N), 2):
        if abs(s[i] - s[j]) < j - i: adj[i].add(j); adj[j].add(i)
    seen, st = {0}, [0]
    while st:
        u = st.pop()
        for v in adj[u] - seen: seen.add(v); st.append(v)
    return len(seen) == N

def analyse(N, s):
    H, A, b = halfspaces(N, s); m, d = A.shape
    x0 = np.array(s[1:], float) - s[0]
    # Chebyshev: max r  s.t.  A x - sqrt2 r >= b
    res = linprog(np.r_[np.zeros(d), -1], A_ub=-np.c_[A, -np.sqrt(2) * np.ones(m)], b_ub=-b,
                  bounds=[(None, None)] * d + [(0, None)])
    inr = np.inf if res.status == 3 else -res.fun
    # acotada: ninguna direccion de recesion (A y >= 0, y != 0)
    bounded = True
    for k in range(d):
        for sgn in (1, -1):
            e = np.zeros(d); e[k] = -sgn
            r = linprog(e, A_ub=-A, b_ub=np.zeros(m), bounds=[(-1, 1)] * d)
            if r.status == 0 and -r.fun > 1e-9: bounded = False
    # facetas: pared h toca si existe x con h=0 y el resto estricto
    facets = 0
    for h in range(m):
        others = [g for g in range(m) if g != h]
        r = linprog(np.r_[np.zeros(d), -1],
                    A_ub=-np.c_[A[others], -np.ones(m - 1)], b_ub=-b[others],
                    A_eq=np.r_[A[h], 0][None], b_eq=[b[h]],
                    bounds=[(None, None)] * d + [(None, 1)])
        if r.status == 0 and -r.fun > 1e-9: facets += 1
    depth = min(abs(s[i] - s[j] - c) for i, j, c in H) / np.sqrt(2)
    return dict(sol=list(s), bounded=bool(bounded), bounded_graph=near_graph_connected(N, s),
                facets=facets, walls=m, inradius=float(inr), depth=float(depth),
                toroidal=all((s[i]-s[j]) % N not in (0, (j-i) % N, (i-j) % N)
                             for i, j in combinations(range(N), 2)))

if __name__ == '__main__':
    out = {}
    for N in range(4, int(sys.argv[1]) + 1 if len(sys.argv) > 1 else 10):
        rows = [analyse(N, s) for s in solutions(N)]
        out[N] = rows
        fb = sum(r['bounded'] for r in rows)
        agree = all(r['bounded'] == r['bounded_graph'] for r in rows)
        f = [r['facets'] for r in rows]; ir = [r['inradius'] for r in rows if np.isfinite(r['inradius'])]
        print(f"N={N:2d} Q={len(rows):4d} acotadas={fb:4d} ({fb/len(rows):.0%}) criterio_grafo_ok={agree} "
              f"facetas min/med/max={min(f)}/{np.mean(f):.1f}/{max(f)} de {rows[0]['walls']}  "
              f"inradio medio={np.mean(ir):.3f}  toroidales={sum(r['toroidal'] for r in rows)}", flush=True)
    json.dump(out, open('chambers.json', 'w'))
