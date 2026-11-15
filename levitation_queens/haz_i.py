"""
Generalizacion de la regla de la clase 3: haz de n paralelas de pendiente a (a^2=-1) + 1 recta B
de pendiente -a.  La paralela j corta a B en la fila r_j y toma las filas r_j + X_j*mu4
(X_j = conjunto no vacio de clases de mu4); B toma el resto.
Condicion (para cada par j,l): ningun x en X_j*mu4, y en X_l*mu4 con x - y en (r_l - r_j)*mu4.
Cuenta orbitas de traslacion = 2 * (#configuraciones con r_1 = 0, ordenadas) / n!
Uso: python haz_i.py n p1 p2 ...     (--clase: calcula ademas la clase minima real, lento)
"""
import sys, itertools
from math import factorial
from collections import Counter
from clase3_regla import datos, toroidal

def configuraciones(p, n):
    g, k, mu4, C, vec = datos(p)
    filas = {X: frozenset(x for i in range(k) if X >> i & 1 for x in C[i]) for X in range(1, 1 << k)}
    def compatibles(X, Y, d):
        bad = {(d * m) % p for m in mu4}
        FY = filas[Y]
        return not any((x - m) % p in FY for x in filas[X] for m in bad)
    comp = {}
    def ok(X, Y, d):
        key = (X, Y, d)
        if key not in comp: comp[key] = compatibles(X, Y, d)
        return comp[key]
    out = []
    def rec(rs, Xs):
        if len(rs) == n: out.append((tuple(rs), tuple(Xs))); return
        for r in range(1, p):
            if r in rs: continue
            for X in range(1, 1 << k):
                if all(ok(Xs[j], X, (r - rs[j]) % p) for j in range(len(rs))):
                    rec(rs + [r], Xs + [X])
    for X in range(1, 1 << k): rec([0], [X])
    return out, (g, k, mu4, C, filas)

def construye(p, a, rs, Xs, filas):
    T = [(-a * r) % p for r in range(p)]                     # B: c = -a r
    for rj, X in zip(rs, Xs):
        for x in filas[X]:
            r = (rj + x) % p
            T[r] = (a * (r - rj) - a * rj) % p               # paralela por (rj, -a rj)
    return T

if __name__ == '__main__':
    n = int(sys.argv[1])
    for p in [int(x) for x in sys.argv[2:] if x.isdigit()]:
        if p % 4 != 1: print(f"p={p}: no hay i"); continue
        cfg, (g, k, mu4, C, filas) = configuraciones(p, n)
        a = next(x for x in mu4 if x not in (1, p - 1))
        assert all(toroidal(construye(p, a, rs, Xs, filas), p) for rs, Xs in cfg[:2000])
        tot = 2 * len(cfg) // factorial(n)
        msg = f"p={p}: haz {n}+1 sobre (i,-i): {len(cfg)} configuraciones ordenadas -> {tot} orbitas de traslacion, {tot*p*p} soluciones"
        if '--clase' in sys.argv:
            from orbitas import min_covers, canon_tr
            cl = Counter(); vistos = set()
            for rs, Xs in cfg:
                for aa in (a, p - a):
                    T = canon_tr(construye(p, aa, rs, Xs, filas), p)
                    if T in vistos: continue
                    vistos.add(T); cl[min_covers(list(T), p)[0]] += 1
            msg += f"   | distintas {len(vistos)}, por clase minima real: {dict(sorted(cl.items()))}"
        print(msg, flush=True)
