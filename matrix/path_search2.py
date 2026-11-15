"""Segunda pasada sobre la componente de 46 (N=8, involución):
  1) ¿Qué tipo de transformaciones son? (reflexiones modulares c -> s-c mod N, otras)
  2) Regla fija de prioridad: en cada paso aplicar la primera T de una lista que lleve a una solución no visitada.
     Longitud alcanzada desde cada inicio -> cuán 'predecible' es recorrer la componente.
  3) Camino hamiltoniano con el mínimo número de generadores distintos (probando los k más aplicables).
Uso: python path_search2.py
"""
import sys, itertools, random
from collections import Counter, defaultdict
import numpy as np
from experiments import G, involution, UF

sys.setrecursionlimit(10000)
N = 8
g = G(N)
mask = involution(g) & g.usable
src, dst, key = g.src[mask], g.dst[mask], g.Tkey[mask]
uf = UF(g.m)
for a, b in zip(src, dst):
    uf.union(int(a), int(b))
lab = uf.labels()
comp = [int(a) for a in np.where(lab == lab[0])[0]]
nbr = defaultdict(dict)
for a, b, k in zip(src, dst, key):
    nbr[int(a)][int(k)] = int(b)


def tperm(k): return [(k // N ** x) % N for x in range(N)]
def cyc(k):
    t = tperm(k)
    return "".join(f"({i} {t[i]})" for i in range(N) if i < t[i]) or "id"


apply_count = Counter(k for a in comp for k in nbr[a])
keys = sorted(apply_count, key=lambda k: (-apply_count[k], k))

# 1) tipo de transformación: reflexión modular completa c -> s - c (mod N) ?
refl = {}
for k in keys:
    t = tperm(k)
    for s in range(N):
        if all(t[c] == (s - c) % N for c in range(N)):
            refl[k] = s
print(f"componente: {len(comp)} nodos, {len(keys)} T distintas")
print(f"T que son reflexión modular c -> s-c (mod {N}): {len(refl)} de {len(keys)}")
for k, s in refl.items():
    print(f"   s={s}: {cyc(k):<22} aplicable en {apply_count[k]} soluciones")
rest = [k for k in keys if k not in refl]
print(f"el resto ({len(rest)}) son transformaciones puntuales; aplicabilidad:",
      dict(Counter(apply_count[k] for k in rest)))
reach_by_refl = sum(1 for a in comp if any(k in refl for k in nbr[a]))
print(f"soluciones con al menos una arista de reflexión modular: {reach_by_refl}/{len(comp)}")


# 2) regla fija de prioridad
def priority_walk(start, order):
    path, seen, a = [start], {start}, start
    seq = []
    while True:
        for k in order:
            b = nbr[a].get(k)
            if b is not None and b not in seen:
                path.append(b); seen.add(b); seq.append(k); a = b
                break
        else:
            return path, seq


def eval_order(order):
    lens = [len(priority_walk(s, order)[0]) for s in comp]
    return max(lens), float(np.mean(lens))


print("\nregla de prioridad fija (primera T de la lista que lleve a una solución nueva):")
print("  orden por aplicabilidad (desc): máx=%d, media=%.1f de %d" % (*eval_order(keys), len(comp)))
random.seed(1)
best = (0, 0.0, keys)
for _ in range(3000):
    o = keys[:]; random.shuffle(o)
    mx, mean = eval_order(o)
    if (mx, mean) > best[:2]:
        best = (mx, mean, o)
print("  mejor de 3000 órdenes aleatorios: máx=%d, media=%.1f" % (best[0], best[1]))
# mejora local sobre el mejor orden
o = best[2][:]
cur = eval_order(o)
for it in range(4000):
    i, j = random.sample(range(len(o)), 2)
    o2 = o[:]; o2[i], o2[j] = o2[j], o2[i]
    r = eval_order(o2)
    if r > cur:
        o, cur = o2, r
print("  tras búsqueda local: máx=%d, media=%.1f" % cur)
best_order = o
mx_start = max(comp, key=lambda s: len(priority_walk(s, best_order)[0]))
p, seq = priority_walk(mx_start, best_order)
print(f"  camino más largo con esa regla: {len(p)} soluciones; T usadas={len(set(seq))}; "
      f"índices en la lista: {[best_order.index(k) for k in seq]}")


# 3) camino hamiltoniano con pocos generadores (los k más aplicables)
def hamiltonian(allowed, limit=400_000):
    allowed = set(allowed)
    adj = {a: [(b, k) for k, b in nbr[a].items() if k in allowed] for a in comp}
    count = [0]
    def dfs(path, seen):
        count[0] += 1
        if count[0] > limit: return None
        if len(path) == len(comp): return list(path)
        cand = [b for b, _ in adj[path[-1]] if b not in seen]
        cand.sort(key=lambda b: sum(1 for c, _ in adj[b] if c not in seen))
        for b in cand:
            path.append(b); seen.add(b)
            r = dfs(path, seen)
            if r: return r
            path.pop(); seen.discard(b)
        return None
    for s in sorted(comp, key=lambda a: len(adj[a])):
        r = dfs([s], {s})
        if r: return r
        if count[0] > limit: return None
    return None


print("\ncamino hamiltoniano (46 soluciones) con los k generadores más aplicables:")
found = None
for k in range(2, len(keys) + 1):
    ham = hamiltonian(keys[:k])
    if ham:
        found = (k, ham); break
if found:
    k, ham = found
    seq = [[kk for kk, bb in nbr[a].items() if bb == b and kk in set(keys[:k])][0] for a, b in zip(ham, ham[1:])]
    used = Counter(seq)
    print(f"  encontrado con k={k} generadores (de {len(keys)}); T usadas en el camino={len(used)}")
    print("  uso de cada generador:", {cyc(kk): c for kk, c in used.most_common()})
    print("  secuencia (posición en la lista de generadores):", [keys.index(kk) for kk in seq])
else:
    print("  no encontrado con búsqueda acotada ni usando los 32 generadores")
