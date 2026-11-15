"""
N = p primo: soluciones toroidales  sigma(i) = a_{clase(i)} * i  con d clases.

clase(i) = (log discreto de i) mod d, para d | p-1  (d = 2: residuo cuadratico o no).

CONDICION EXACTA. Sea L(x) = clase(x). La solucion es toroidal  <=>  las tres aplicaciones
    j -> j + L(a_j),    j -> j + L(a_j + 1),    j -> j + L(a_j - 1)        (mod d)
son permutaciones de Z_d.  (Para d = 2: a y b del mismo tipo en los tres sentidos.)
Es decir: elegir las pendientes es resolver un problema "de reinas" en Z_d cuyas piezas
son los valores de a, clasificados por su TIPO  (L(a), L(a+1), L(a-1)).

Con eso se cuentan y se generan sin fuerza bruta: se recorren las ternas de permutaciones
de Z_d y se multiplican cuantos a hay de cada tipo.

Uso: python clases_primos.py 13 17 19 23 29 31 37 41 43 [--d 2 3 4 6]
"""
import argparse, itertools
from collections import Counter, defaultdict


def tor(s):
    N = len(s)
    return len(set(s)) == N and len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N


def dlog_class(p, d):
    g = next(g for g in range(2, p) if len({pow(g, e, p) for e in range(p - 1)}) == p - 1)
    idx = [None] * p
    x = 1
    for e in range(p - 1):
        idx[x] = e % d
        x = x * g % p
    return idx


def types(p, d):
    """tipo de cada a en 2..p-2: (L(a), L(a+1), L(a-1)) -> lista de a"""
    L = dlog_class(p, d)
    T = defaultdict(list)
    for a in range(2, p - 1):
        T[(L[a], L[a + 1], L[a - 1])].append(a)
    return L, T


def solutions_by_classes(p, d, limit=None):
    """genera las tuplas de pendientes (a_0..a_{d-1}) validas, sin fuerza bruta"""
    L, T = types(p, d)
    out = []

    def rec(j, used0, usedp, usedm, cur):
        if j == d:
            out.append(tuple(cur)); return
        for (t0, tp, tm), As in T.items():
            x0, xp, xm = (j + t0) % d, (j + tp) % d, (j + tm) % d
            if x0 in used0 or xp in usedp or xm in usedm:
                continue
            for a in As:
                cur.append(a)
                rec(j + 1, used0 | {x0}, usedp | {xp}, usedm | {xm}, cur)
                cur.pop()
                if limit and len(out) >= limit:
                    return
    rec(0, frozenset(), frozenset(), frozenset(), [])
    return L, out


def count_by_classes(p, d):
    """solo el numero (programacion dinamica sobre los tipos)"""
    L, T = types(p, d)
    n = {t: len(v) for t, v in T.items()}
    from functools import lru_cache

    @lru_cache(None)
    def rec(j, u0, up, um):
        if j == d:
            return 1
        tot = 0
        for (t0, tp, tm), c in n.items():
            x0, xp, xm = (j + t0) % d, (j + tp) % d, (j + tm) % d
            if u0 >> x0 & 1 or up >> xp & 1 or um >> xm & 1:
                continue
            tot += c * rec(j + 1, u0 | 1 << x0, up | 1 << xp, um | 1 << xm)
        return tot
    return rec(0, 0, 0, 0)


def to_solution(p, d, L, slopes):
    return tuple(slopes[L[i]] * i % p if i else 0 for i in range(p))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('ps', type=int, nargs='+')
    ap.add_argument('--d', type=int, nargs='+', default=[1, 2, 3, 4, 6])
    a = ap.parse_args()
    print("soluciones toroidales por (0,0) con d clases (incluye las de menos clases); entre parentesis, tipos de a: {tipo: cuantos}")
    for p in a.ps:
        row = []
        for d in a.d:
            if (p - 1) % d:
                row.append(f"d={d}: -")
                continue
            row.append(f"d={d}: {count_by_classes(p, d)}")
        _, T2 = types(p, 2)
        n2 = sorted(len(v) for v in T2.values())
        pairs = sum(n * (n - 1) for n in n2)
        print(f"p={p:2d} (p-1={p-1}): " + "  ".join(row) + f"   | d=2: tipos {n2} -> pares distintos = suma n(n-1) = {pairs}")
