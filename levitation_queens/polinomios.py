"""
Cada toroidal como polinomio T(x) sobre Z_p: grado y numero de terminos.

Para cada solucion distinta (data/orbitas_p{p}.json) se prueban las p traslaciones del origen y los
8 giros, y se guarda la forma con menos terminos. El grado no cambia al trasladar ni al escalar.
Rectas: grado 1. Dos espirales con k bloques desde el cruce: exponentes 1 + j*(p-1)/k.

Uso: ../venv/bin/python polinomios.py 17 19
"""
import sys, json
import numpy as np
from orbitas import mats, apply


def coefs(T, p):
    """coeficientes c[0..p-1] de T(x) = sum c[k] x^k (interpolacion en Z_p)."""
    c = [0] * p
    c[0] = T[0] % p
    for k in range(1, p):
        # c_k = -sum_{a != 0} T(a) a^{-k}   (k < p-1);  c_{p-1} = -sum_a T(a)
        if k < p - 1:
            c[k] = -sum(T[a] * pow(a, p - 1 - k, p) for a in range(1, p)) % p
        else:
            c[k] = -sum(T) % p
    return c


def mejor(T, p, D):
    best = None
    for M in D:
        U = [int(x) for x in apply(M, T, p)]
        for s in range(p):
            V = [(U[(r + s) % p] - U[s]) % p for r in range(p)]
            c = coefs(V, p)
            ex = [k for k in range(p) if c[k]]
            key = (len(ex), max(ex))
            if best is None or key < best[0]:
                best = (key, ex, [c[k] for k in ex])
    return best


if __name__ == "__main__":
    for p in map(int, sys.argv[1:]):
        D = mats(p)[::p - 1]
        print(f"p={p}")
        for o in sorted(json.load(open(f"data/orbitas_p{p}.json")), key=lambda o: (o["clase"], o["tam"])):
            T = o["rep"]
            grados = sorted({max(k for k, x in enumerate(coefs([int(v) for v in apply(M, T, p)], p)) if x) for M in D})
            (nt, _), ex, cf = mejor(T, p, D)
            print(f"  clase {o['clase']} tam {o['tam']:3d} sim {o['stab']:2d}: grado {grados}  "
                  f"minimo {nt} terminos, exponentes {ex}")
