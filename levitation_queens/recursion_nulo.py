"""Modelo nulo de recursion_clases: permutaciones al azar (NO soluciones) frente a las soluciones de
clase < m. Si tambien se "construyen" con 2 o 3 piezas, la prueba no dice nada."""
import sys
import numpy as np
from recursion_clases import load

p = int(sys.argv[1]) if len(sys.argv) > 1 else 13
cls, S = load(p)
full = (1 << p) - 1
W = (1 << np.arange(p)).astype(np.int64)
rng = np.random.default_rng(1)
for m in range(3, cls.max() + 1):
    low = S[cls < m]
    b_ok = c_ok = 0; n = 500
    for _ in range(n):
        t = rng.permutation(p)
        ms = np.unique(((low == t) * W).sum(1)); ms = [int(x) for x in ms if x]
        mx = [x for x in ms if not any(y != x and (x & ~y) == 0 for y in ms)]
        b = any((full & ~(x | y)) == 0 for i, x in enumerate(mx) for y in mx[i:])
        c = b or any((full & ~(x | y | z)) == 0 for i, x in enumerate(mx) for j, y in enumerate(mx[i:], i) for z in mx[j:])
        b_ok += b; c_ok += c
    print(f"p={p} azar contra clases < {m} ({len(low)} piezas): dos piezas {100*b_ok/n:.1f}% | hasta tres {100*c_ok/n:.1f}%", flush=True)
