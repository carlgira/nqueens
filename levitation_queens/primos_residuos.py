"""
N primo: soluciones toroidales por CLASES. sigma(i) = a_{clase(i)} * i  (mod p), donde la
clase de i es su coset en Z_p^* / (potencias d-esimas):
    d = 1: una pendiente  (toroidal lineal, a*i)
    d = 2: dos pendientes: a si i es residuo cuadratico, b si no lo es
    d = 3: tres pendientes segun la clase cubica (solo si 3 | p-1)
y despues todas las traslaciones  sigma(i + c) + b.

Caso particular d = 2, b = -a:  sigma(i) = a * i^((p+1)/2), valido sii p = 1 (mod 4),
con exactamente (p-5)/2 valores de a.

Uso: python primos_residuos.py 13 17 19 23 29 31 37 41 43
"""
import itertools, sys


def tor(s):
    N = len(s)
    return len(set(s)) == N and len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N


def cosets(p, d):
    g = next(g for g in range(2, p) if len({pow(g, e, p) for e in range(p - 1)}) == p - 1)
    idx = [-1] * p
    x = 1
    for e in range(p - 1):
        idx[x] = e % d
        x = x * g % p
    return idx


def base(p, d):
    """soluciones toroidales por (0,0) con d pendientes"""
    idx = cosets(p, d)
    out = {}
    for sl in itertools.product(range(1, p), repeat=d):
        s = tuple(sl[idx[i]] * i % p if i else 0 for i in range(p))
        if tor(s):
            out[s] = sl
    return out


def translations(S, p):
    full = set()
    for s in S:
        for c in range(p):
            sh = s[c:] + s[:c]
            for b in range(p):
                full.add(tuple((x + b) % p for x in sh))
    return full


if __name__ == '__main__':
    print(f"{'p':>3} {'mod4':>4} | lineales p(p-3) | dos pendientes: pares (a,b), con b=-a | soluciones distintas con traslaciones (1 y 2 pendientes)")
    for p in [int(a) for a in sys.argv[1:]] or [7, 11, 13, 17, 19, 23, 29, 31]:
        B1, B2 = base(p, 1), base(p, 2)
        pairs = [sl for sl in B2.values() if sl[0] != sl[1]]
        neg = sum((a + b) % p == 0 for a, b in pairs)
        full = translations(set(B1) | set(B2), p) if p <= 43 else None
        print(f"{p:3d} {p%4:4d} | {p*(p-3):6d} | {len(pairs):4d} pares, {neg} con b=-a [(p-5)/2 = {(p-5)//2 if p%4==1 else '-'}] | "
              f"{len(full) if full else '?'}", flush=True)
