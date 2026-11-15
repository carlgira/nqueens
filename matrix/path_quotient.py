"""¿El reflejo de filas q -> q[::-1] es un automorfismo del grafo etiquetado de 46 nodos?
Las T actúan sobre las COLUMNAS (valores de q); el reflejo de filas actúa sobre los ÍNDICES -> conmutan.
Si es así, el grafo de 46 es el doble de un grafo cociente de 23 nodos, y una palabra de T aplicada a a
y a rho(a) da caminos idénticos en T. Se busca el camino hamiltoniano del cociente con pocos generadores.
Uso: python path_quotient.py
"""
import sys
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
cset = set(comp)
nbr = defaultdict(dict)
for a, b, k in zip(src, dst, key):
    nbr[int(a)][int(k)] = int(b)

index = {tuple(s): i for i, s in enumerate(g.sols.tolist())}
rho = {a: index[tuple(g.sols[a][::-1].tolist())] for a in comp}
print("componente cerrada bajo reflejo de filas:", all(rho[a] in cset for a in comp))
print("puntos fijos del reflejo:", sum(1 for a in comp if rho[a] == a))
ok = all(nbr[rho[a]].get(k) == rho[b] for a in comp for k, b in nbr[a].items())
print("el reflejo conserva TODAS las aristas con su etiqueta T:", ok)

# cociente
cls = {a: min(a, rho[a]) for a in comp}
reps = sorted(set(cls.values()))
print(f"nodos del cociente: {len(reps)}")
qn = {c: {k: cls[b] for k, b in nbr[c].items()} for c in reps}
cnt = Counter(k for c in reps for k in qn[c])
print(f"T distintas: {len(cnt)}; aplicabilidad en clases:", dict(sorted(Counter(cnt.values()).items(), reverse=True)))
deg = Counter(len(qn[c]) for c in reps)
print("grado de cada clase:", dict(sorted(deg.items())))
loops = sum(1 for c in reps for k, d in qn[c].items() if d == c)
print("aristas que unen una clase consigo misma:", loops)


def tperm(k): return [(k // N ** x) % N for x in range(N)]
def cyc(k):
    t = tperm(k)
    return "".join(f"({i} {t[i]})" for i in range(N) if i < t[i]) or "id"


keys = sorted(cnt, key=lambda k: (-cnt[k], k))


def hamiltonian(allowed, limit=3_000_000):
    allowed = set(allowed)
    adj = {c: [(d, k) for k, d in qn[c].items() if k in allowed and d != c] for c in reps}
    count = [0]
    def dfs(path, seen):
        count[0] += 1
        if count[0] > limit: return None
        if len(path) == len(reps): return list(path)
        cand = [d for d, _ in adj[path[-1]] if d not in seen]
        cand.sort(key=lambda d: sum(1 for e, _ in adj[d] if e not in seen))
        for d in cand:
            path.append(d); seen.add(d)
            r = dfs(path, seen)
            if r: return r
            path.pop(); seen.discard(d)
        return None
    for s in sorted(reps, key=lambda c: len(adj[c])):
        r = dfs([s], {s})
        if r: return r
        if count[0] > limit: return None
    return None


print("\ncamino hamiltoniano en el cociente (23 clases) con los k generadores más aplicables:")
for k in range(2, len(keys) + 1):
    ham = hamiltonian(keys[:k])
    if ham:
        seq = [[kk for kk, d in qn[a].items() if d == b and kk in set(keys[:k])][0] for a, b in zip(ham, ham[1:])]
        used = Counter(seq)
        print(f"  encontrado con k={k}; T usadas={len(used)}; longitud={len(ham)} clases")
        print("  T usadas:", {cyc(kk): c for kk, c in used.most_common()})
        print("  secuencia (posición en la lista):", [keys.index(kk) for kk in seq])
        # levantado al grafo de 46: aplicar la palabra desde un representante real
        a = ham[0]; path = [a]
        for kk in seq:
            a = nbr[a][kk]; path.append(a)
        print(f"  levantado: {len(path)} soluciones distintas = {len(set(path))}, "
              f"y su imagen por el reflejo de filas usa la MISMA palabra de T")
        break
else:
    print("  no encontrado")
