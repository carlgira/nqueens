"""Enumeración EXHAUSTIVA de los caminos hamiltonianos del cociente (23 clases, N=8, involución).
Cada camino = secuencia de 22 transformaciones T (la etiqueta de cada paso).
Calcula: cuántos son, mínimo exacto de T distintas por camino, y los patrones que se repiten.
Uso: python ham_enum.py
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
nbr = defaultdict(dict)
for a, b, k in zip(src, dst, key):
    nbr[int(a)][int(k)] = int(b)
index = {tuple(s): i for i, s in enumerate(g.sols.tolist())}
rho = {a: index[tuple(g.sols[a][::-1].tolist())] for a in comp}
cls = {a: min(a, rho[a]) for a in comp}
reps = sorted(set(cls.values()))
qn = {c: {k: cls[b] for k, b in nbr[c].items()} for c in reps}


def tperm(k): return [(k // N ** x) % N for x in range(N)]
def cyc(k):
    t = tperm(k)
    return "".join(f"({i} {t[i]})" for i in range(N) if i < t[i]) or "id"


paths = []                                      # cada elemento: (clase_inicial, [T1..T22])
def dfs(c, seen, seq, start):
    if len(seen) == len(reps):
        paths.append((start, tuple(seq))); return
    for k, d in qn[c].items():
        if d not in seen:
            seen.add(d); seq.append(k)
            dfs(d, seen, seq, start)
            seq.pop(); seen.discard(d)

for s in reps:
    dfs(s, {s}, [], s)

print(f"caminos hamiltonianos (con etiqueta T en cada paso): {len(paths)}")
distinct = Counter(len(set(seq)) for _, seq in paths)
print("T distintas por camino (mínimo exacto = %d):" % min(distinct), dict(sorted(distinct.items())))

# T más usadas en el conjunto de caminos
use = Counter(k for _, seq in paths for k in seq)
print("\nT más frecuentes en todos los pasos de todos los caminos:")
for k, c in use.most_common(5):
    print(f"   {cyc(k):<22} {c:>7} pasos ({100*c/(len(paths)*22):.1f}%)")

# pares consecutivos de T que más se repiten
bi = Counter((a, b) for _, seq in paths for a, b in zip(seq, seq[1:]))
exp_total = sum(bi.values())
print(f"\npares consecutivos distintos de T: {len(bi)}  (posibles: {len(use)**2})")
print("más repetidos:")
for (a, b), c in bi.most_common(5):
    print(f"   {cyc(a)} -> {cyc(b)}: {c} veces ({100*c/exp_total:.2f}% de los pasos)")

# ¿algún camino repite una subpalabra de T dentro de sí mismo? (longitud máxima)
best = (0, None)
for s, seq in paths:
    n = len(seq)
    for L in range(n // 2, 1, -1):
        found = False
        seen = {}
        for i in range(n - L + 1):
            w = seq[i:i + L]
            if w in seen:
                found = True; break
            seen[w] = i
        if found:
            if L > best[0]:
                best = (L, (s, seq))
            break
L, (s, seq) = best
print(f"\nsubpalabra de T más larga que se repite DENTRO de un mismo camino: {L} pasos")
print("   (de un total de 22 pasos por camino)")
# camino con menos T distintas
mn = min(paths, key=lambda p: len(set(p[1])))
print(f"\ncamino con menos T distintas ({len(set(mn[1]))}):")
print("   uso:", {cyc(k): c for k, c in Counter(mn[1]).most_common()})
