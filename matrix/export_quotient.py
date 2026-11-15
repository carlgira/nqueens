"""Exporta el cociente de 23 clases (N=8, involución) como lista de adyacencia con multiplicidades,
para contar caminos hamiltonianos exactamente con un programa en C.
Uso: python export_quotient.py > quotient.txt
"""
from collections import defaultdict, Counter
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
comp = [int(a) for a in np.where(lab == lab[0])[0]]
nbr = defaultdict(dict)
for a, b, k in zip(src, dst, key):
    nbr[int(a)][int(k)] = int(b)
index = {tuple(s): i for i, s in enumerate(g.sols.tolist())}
rho = {a: index[tuple(g.sols[a][::-1].tolist())] for a in comp}
cls = {a: min(a, rho[a]) for a in comp}
reps = sorted(set(cls.values()))
pos = {c: i for i, c in enumerate(reps)}
mult = Counter()                       # (clase_i, clase_j) -> nº de T distintas que las unen
for c in reps:
    for k, b in nbr[c].items():
        mult[(pos[c], pos[cls[b]])] += 1
print(len(reps))
for (i, j), m in sorted(mult.items()):
    print(i, j, m)
