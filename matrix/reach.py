"""Alcance desde una solución con las 3 reglas (intercambio, ciclo de 3, doble intercambio), y paridad de las 8 imágenes D4.
Uso: python reach.py
"""
import itertools, random, sys
from collections import Counter, deque
import numpy as np
from build_graph import solve, sym_images


def parity(q):
    n = len(q); seen = [False] * n; c = 0
    for i in range(n):
        if not seen[i]:
            c += 1; j = i
            while not seen[j]:
                seen[j] = True; j = q[j]
    return (n - c) & 1


def moves(n):
    """Cada movimiento = (tipo, permutación de posiciones). Aplicar a la lista q es equivalente a una T sobre columnas."""
    out = []
    for i, j in itertools.combinations(range(n), 2):
        out.append(("interc", {i: j, j: i}))
    for i, j, k in itertools.combinations(range(n), 3):
        out.append(("ciclo3", {i: j, j: k, k: i}))
        out.append(("ciclo3", {i: k, k: j, j: i}))
    for i, j, k, l in itertools.combinations(range(n), 4):
        out.append(("doble", {i: j, j: i, k: l, l: k}))
        out.append(("doble", {i: k, k: i, j: l, l: j}))
        out.append(("doble", {i: l, l: i, j: k, k: j}))
    return out


def build(n):
    S = [tuple(int(x) for x in s) for s in solve(n)]
    idx = {s: i for i, s in enumerate(S)}
    mv = moves(n)
    adj = [[] for _ in S]
    for a, q in enumerate(S):
        imgs = {tuple(int(v) for v in im) for im in sym_images(q)}
        for t, perm in mv:
            r = list(q)
            for dst, src in perm.items():
                r[dst] = q[src]
            r = tuple(r)
            b = idx.get(r)
            if b is not None and r not in imgs:
                adj[a].append((b, t))
    return S, adj


def bfs(adj, s, allowed):
    dist = {s: 0}; dq = deque([s])
    while dq:
        a = dq.popleft()
        for b, t in adj[a]:
            if t in allowed and b not in dist:
                dist[b] = dist[a] + 1; dq.append(b)
    return dist


random.seed(5)
print("=== Paridad de las 8 imágenes D4 de una solución (cuántas son pares) ===")
for n in range(8, 14):
    S = [tuple(int(x) for x in s) for s in solve(n)]
    pats = Counter()
    for q in random.sample(S, min(300, len(S))):
        par = tuple(parity(tuple(int(v) for v in im)) for im in sym_images(q))
        pats[sum(1 for p in par if p == par[0])] += 1      # nº de imágenes con la misma paridad que q
    print(f"  N={n:2d} (N mod 4 = {n % 4}): imágenes con la misma paridad que la solución -> {dict(pats)}")

print("\n=== Alcance desde una solución al azar ===")
for n, starts in ((8, 8), (9, 8), (10, 8), (11, 6), (12, 4)):
    S, adj = build(n)
    m = len(S)
    rng = random.Random(n)
    for label, allowed in (("las 3 reglas", {"interc", "ciclo3", "doble"}),
                           ("solo ciclo3+doble (conserva paridad)", {"ciclo3", "doble"})):
        res = []
        for s in rng.sample(range(m), starts):
            d = bfs(adj, s, allowed)
            lv = sorted(d.values())
            tot = len(lv)
            if tot == 1:
                res.append((1, 0, 0, 0)); continue
            q50 = lv[int(0.5 * tot) - 1]; q90 = lv[int(0.9 * tot) - 1]
            res.append((tot, max(lv), q50, q90))
        reached = [r[0] for r in res]
        print(f"  N={n:2d} {label:<40} alcanza (de {m}): " +
              " ".join(f"{r[0]:>5}" for r in res) + f"   | pasos máx/50%/90% del mayor: {max(res)[1]}/{max(res)[2]}/{max(res)[3]}")
    sys.stdout.flush()
