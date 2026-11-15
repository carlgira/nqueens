"""
Toroidales: ¿sus rectas pasan todas por una misma reina (centro)?
Clase radial = min sobre las reinas r0 del n de pendientes distintas g(r) = (T(r)-T(r0))/(r-r0) mod p.
Si las rectas de una cobertura minima son concurrentes, clase radial = clase (min rectas).
Uso: python radial.py 13 17 19
"""
import sys
from collections import Counter
import numpy as np

for p in [int(x) for x in sys.argv[1:]] or [13, 17]:
    D = np.loadtxt(f'data/clases_toro_p{p}.txt', dtype=np.int64)
    cls, S = D[:, 0], D[:, 1:]
    inv = np.array([0] + [pow(x, -1, p) for x in range(1, p)])
    best = np.full(len(S), 99)
    for r0 in range(p):
        r = np.array([x for x in range(p) if x != r0])
        g = ((S[:, r] - S[:, [r0]]) * inv[(r - r0) % p]) % p
        gs = np.sort(g, axis=1)
        nd = 1 + (gs[:, 1:] != gs[:, :-1]).sum(1)
        best = np.minimum(best, nd)
    tab = Counter(zip(cls.tolist(), best.tolist()))
    print(f"p={p}: (clase por rectas, clase radial) -> soluciones")
    for k in sorted(set(cls.tolist())):
        print(f"   clase {k}: " + "  ".join(f"radial {b}: {n}" for (c, b), n in sorted(tab.items()) if c == k))
