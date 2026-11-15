"""
Familia "radial impar": todas las rectas pasan por una reina (el origen) y T(-r) = -T(r).
T(r) = a_j * r para r en la clase {+g^j, -g^j}, j = 0..(p-3)/2  (una pendiente por pareja de filas opuestas).
Cuenta las toroidales de este tipo segun el numero de pendientes distintas usadas.
Cada una da p^2 soluciones (traslaciones). Uso: python radial_impar.py 13 17 19 23
"""
import sys
from collections import Counter

def cuenta(p, guardar=False):
    g = next(g for g in range(2, p) if len({pow(g, e, p) for e in range(p - 1)}) == p - 1)
    h = (p - 1) // 2
    idx = {}
    for j in range(h):
        x = pow(g, j, p); idx[x] = j; idx[p - x] = j
    # tabla: clase de a*g^j para a = pendiente, y de (a+1), (a-1)
    slopes = list(range(2, p - 1))
    tab = [[(1 << idx[a * pow(g, j, p) % p], 1 << idx[(a + 1) * pow(g, j, p) % p], 1 << idx[(a - 1) * pow(g, j, p) % p]) for a in slopes] for j in range(h)]
    res = Counter(); sols = []
    ch = [0] * h
    def rec(j, c, s, d):
        if j == h:
            res[len(set(ch))] += 1
            if guardar: sols.append(tuple(ch))
            return
        for i, (x, y, z) in enumerate(tab[j]):
            if c & x or s & y or d & z: continue
            ch[j] = slopes[i]; rec(j + 1, c | x, s | y, d | z)
    rec(0, 0, 0, 0)
    return res, sols, g

if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]]:
        res, _, g = cuenta(p)
        print(f"p={p}: toroidales radiales impares por numero de pendientes: {dict(sorted(res.items()))}  total {sum(res.values())}", flush=True)
