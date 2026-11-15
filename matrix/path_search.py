"""Caminos 'predecibles' dentro de la componente de 46 soluciones (N=8, criterio involución).

En esta componente cada arista a--b lleva una T involutiva (T = P_b P_a^T = T^-1), que actúa sobre las
COLUMNAS: q'(i) = T[q(i)]. Se escribe como intercambios de columnas, p.ej. (0 3)(1 5).
Un camino es 'predecible' si la secuencia de T que aplica sigue una regla corta:
  A) palabra periódica de longitud p (alternar T1,T2,... repetido)
  B) pocos generadores distintos en total (camino hamiltoniano con k transformaciones)
Uso: python path_search.py
"""
import time
from collections import Counter, defaultdict
import numpy as np
from experiments import G, involution, UF

N = 8
g = G(N)
mask = involution(g) & g.usable
src, dst, key = g.src[mask], g.dst[mask], g.Tkey[mask]
uf = UF(g.m)
for a, b in zip(src, dst):
    uf.union(int(a), int(b))
lab = uf.labels()
comp = np.where(lab == lab[0])[0]
print(f"componente de la solución 0: {len(comp)} nodos")

nbr = defaultdict(dict)                       # nbr[a][T] = b  (aplicar T a la solución a da b)
for a, b, k in zip(src, dst, key):
    nbr[int(a)][int(k)] = int(b)


def tperm(k):
    return [(k // N ** x) % N for x in range(N)]


def cyc(k):
    t = tperm(k)
    s = "".join(f"({i} {t[i]})" for i in range(N) if i < t[i])
    return s or "id"


apply_count = Counter(k for a in comp for k in nbr[int(a)])
keys = sorted(apply_count, key=lambda k: -apply_count[k])
print(f"transformaciones T distintas en la componente: {len(keys)}")
print("aplicabilidad (a cuántas de las 46 soluciones se puede aplicar):",
      dict(sorted(Counter(apply_count.values()).items(), reverse=True)))
print("más aplicables:")
for k in keys[:6]:
    print(f"   {cyc(k):<22} en {apply_count[k]} soluciones")


def follow(a, word):
    path, seen, i = [a], {a}, 0
    while True:
        b = nbr[a].get(word[i % len(word)])
        if b is None or b in seen:
            return path
        path.append(b); seen.add(b); a = b; i += 1


def best_for_words(words):
    best = []
    for w in words:
        for a in comp:
            p = follow(int(a), w)
            if len(p) > len(best):
                best = p; bw = w
    return len(best), bw, best


# ---- A) palabras periódicas
t0 = time.time()
for p, pool in ((2, keys), (3, keys[:30]), (4, keys[:18])):
    import itertools
    words = [w for w in itertools.product(pool, repeat=p) if all(w[i] != w[(i + 1) % p] for i in range(p))]
    L, w, path = best_for_words(words)
    print(f"\npalabra periódica de longitud {p}: camino más largo = {L} soluciones "
          f"({len(words)} palabras probadas, {time.time()-t0:.0f}s)")
    print("   palabra:", " , ".join(cyc(k) for k in w))

# ---- B) conjunto mínimo de generadores que conecta la componente (voraz) y camino hamiltoniano
def components_with(gens):
    u = UF(g.m)
    for a in comp:
        for k in gens:
            b = nbr[int(a)].get(k)
            if b is not None:
                u.union(int(a), b)
    return len({u.find(int(a)) for a in comp})


gens, remaining = [], list(keys)
while components_with(gens) > 1 and remaining:
    best = min(remaining, key=lambda k: components_with(gens + [k]))
    gens.append(best); remaining.remove(best)
print(f"\ngeneradores (voraz) para que las 46 queden conectadas: {len(gens)}")
for k in gens:
    print("   ", cyc(k), f"(aplicable en {apply_count[k]})")


def hamiltonian(gens, limit=2_000_000):
    allowed = set(gens)
    nodes = [int(a) for a in comp]
    adj = {a: [(b, k) for k, b in nbr[a].items() if k in allowed] for a in nodes}
    sys_nodes = [0]
    count = [0]
    def dfs(path, seen):
        count[0] += 1
        if count[0] > limit:
            return None
        if len(path) == len(nodes):
            return list(path)
        cur = path[-1]
        cand = [b for b, _ in adj[cur] if b not in seen]
        cand.sort(key=lambda b: sum(1 for c, _ in adj[b] if c not in seen))   # Warnsdorff
        for b in cand:
            path.append(b); seen.add(b)
            r = dfs(path, seen)
            if r:
                return r
            path.pop(); seen.discard(b)
        return None
    for s in sorted(nodes, key=lambda a: len(adj[a])):
        r = dfs([s], {s})
        if r:
            return r
    return None


import sys
sys.setrecursionlimit(10000)
ham = hamiltonian(gens)
print("\ncamino hamiltoniano usando solo esos generadores:", "ENCONTRADO" if ham else "no encontrado (límite de búsqueda)")
if ham:
    ts = []
    for a, b in zip(ham, ham[1:]):
        ts.append([k for k, bb in nbr[a].items() if bb == b and k in set(gens)][0])
    print("   secuencia de transformaciones (índice del generador):", [gens.index(k) for k in ts])
