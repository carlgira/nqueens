"""
Primer nivel de fusion entre toroidales de un tramo, con todo numerado.

Toroidal (a, c):  L(i) = a*i + c  (mod p).   Fusion: T = (a1, c1) en las filas abiertas de la
plantilla + (a2, c2) en las cerradas. Plantilla de onda (u, x, w): filas con (u*i - x) mod p < w.

Para cada primo se enumeran todas las fusiones validas y se agrupan por el par de pendientes
(con signo, en (-p/2, p/2]) y por la relacion entre ellas:
    opuestas   a2 = -a1        inversas  a2 = 1/a1        inv. opuestas  a2 = -1/a1
¿Se repiten los mismos pares y recuentos de un primo a otro?

Uso: python nivel1.py 11 13 17 19 23
"""
import sys, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def sg(x, p):
    x %= p
    return x if x <= p // 2 else x - p


def gates(p):
    out = {}
    for u in range(1, p // 2 + 1):
        for x in range(p):
            for w in range(1, p):
                G = frozenset(i for i in range(p) if (u * i - x) % p < w)
                out.setdefault(G, (u, x, w))
    return out


def level1(p):
    slopes = list(range(2, p - 1))
    full = (1 << p) - 1
    res = {}                                   # solucion -> (a1, c1, a2, c2, u, x, w) con menos filas cambiadas
    for G, (u, x, w) in gates(p).items():
        Gs = sorted(G); Cs = [i for i in range(p) if i not in G]
        if len(Gs) < len(Cs):
            continue                           # se toma G como el lado grande (la toroidal "base")
        index = defaultdict(list)
        for a in slopes:
            for c in range(p):
                cm = d1 = d2 = 0
                for i in Cs:
                    v = (a * i + c) % p
                    cm |= 1 << v; d1 |= 1 << (i + v); d2 |= 1 << (i - v + p)
                index[cm].append((d1, d2, a, c))
        for a in slopes:
            for c in range(p):
                cg = e1 = e2 = 0
                for i in Gs:
                    v = (a * i + c) % p
                    cg |= 1 << v; e1 |= 1 << (i + v); e2 |= 1 << (i - v + p)
                for d1, d2, a2, c2 in index.get(full ^ cg, ()):
                    if a2 == a or (e1 & d1) or (e2 & d2):
                        continue
                    t = tuple((a * i + c) % p if i in G else (a2 * i + c2) % p for i in range(p))
                    if t not in res or len(Cs) < res[t][7]:
                        res[t] = (a, c, a2, c2, u, x, w, len(Cs))
    return res


def relation(a1, a2, p):
    if (a1 + a2) % p == 0:
        return 'opuestas'
    if (a1 * a2) % p == 1:
        return 'inversas'
    if (a1 * a2) % p == p - 1:
        return 'inv. opuestas'
    return 'otra'


if __name__ == '__main__':
    out = {}
    for p in [int(x) for x in sys.argv[1:]] or [11, 13, 17]:
        R = level1(p)
        rel = Counter(relation(v[0], v[2], p) for v in R.values())
        pairs = Counter((sg(v[0], p), sg(v[2], p)) for v in R.values())
        size = Counter(v[7] for v in R.values())
        freq = Counter(v[4] for v in R.values())
        print(f"===== p={p}: {len(R)} soluciones de nivel 1 (fusion de dos toroidales de un tramo)")
        print(f"   relacion entre pendientes: {dict(rel)}")
        print(f"   filas que vienen de la segunda toroidal: {dict(sorted(size.items()))}")
        print(f"   frecuencia de la plantilla: {dict(sorted(freq.items()))}")
        top = sorted(pairs.items(), key=lambda kv: (abs(kv[0][0]), kv[0]))
        print("   pares (a1, a2) -> soluciones (solo a1 > 0): " + "  ".join(f"({a},{b}):{n}" for (a, b), n in top if a > 0))
        out[p] = {f'{a},{b}': n for (a, b), n in pairs.items()}
    json.dump(out, open(os.path.join(HERE, 'data', 'nivel1_pares.json'), 'w'))
