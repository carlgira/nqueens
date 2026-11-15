"""
N = p primo. Familia "escalera simetrica" (no toroidales de 2 alfas):

    alfa dominante A, alfa raro b = A + delta, con el escalon FORZADO por
            q * A = -2 * delta   (mod p)        <=>   delta = -q*A/2
    k escalones (k >= 2), separados q pasos, colocados simetricamente:
            posiciones (indice de paso)  r0, r0+q, ..., r0+(k-1)q,   r0 = (p-2-(k-1)q)/2
    y columna inicial c.

k = 2 son "dos escalones simetricos" (con r0 = 0: los extremos desplazados);
k >= 3 la escalera regular.

Uso: python escaleras.py 7 11 13 17 19 23 29 31 37 41 43 47
"""
import json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def valid(sol):
    N = len(sol)
    return len(set(sol)) == N and len({i + c for i, c in enumerate(sol)}) == N and len({i - c for i, c in enumerate(sol)}) == N


def build(p, A, q, k, c):
    """-> solucion (o None si la geometria no cabe)"""
    span = (k - 1) * q
    if (p - 2 - span) < 0 or (p - 2 - span) % 2:
        return None
    r0 = (p - 2 - span) // 2
    delta = (-q * A * pow(2, p - 2, p)) % p
    if delta == 0:
        return None
    jumps = {r0 + t * q for t in range(k)}
    sol = [c % p]
    for step in range(p - 1):
        sol.append((sol[-1] + A + (delta if step in jumps else 0)) % p)
    return tuple(sol)


def family(p):
    out = {}
    for A in range(1, p):
        for q in range(1, p - 1):
            for k in range(2, p):
                if (k - 1) * q > p - 2:
                    break
                for c in range(p):
                    s = build(p, A, q, k, c)
                    if s is not None and valid(s) and len({(s[i + 1] - s[i]) % p for i in range(p - 1)}) == 2:
                        out[s] = (A, q, k)        # (si todos los pasos son escalon queda 1 alfa: toroidal, se descarta)
    return out


if __name__ == '__main__':
    cat = {}
    f = os.path.join(HERE, 'data', 'pocos_alfas_D2.json')
    if os.path.exists(f):
        cat = json.load(open(f))
    print(f"{'p':>3} | familia | k=2 (dos escalones) | k>=3 (escalera) | catalogo: no toroidales de 2 alfas | de ellas en la familia")
    for p in [int(x) for x in sys.argv[1:]] or [7, 11, 13, 17, 19, 23, 29, 31, 37]:
        F = family(p)
        k2 = sum(v[2] == 2 for v in F.values())
        extra = ''
        if str(p) in cat:
            def tor(s):
                return len({(i + c) % p for i, c in enumerate(s)}) == p and len({(i - c) % p for i, c in enumerate(s)}) == p
            C = {tuple(s) for s, u in cat[str(p)] if len(u) == 2 and not tor(s)}
            extra = f"{len(C):5d} | {len(C & set(F)):5d}   (familia fuera del catalogo: {len(set(F) - C)})"
        print(f"{p:3d} | {len(F):6d} | {k2:6d} | {len(F)-k2:6d} | {extra}", flush=True)


# ---------------------------------------------------------------- dos espaciados alternos
def build2(p, A, g1, g2, k, f_num, f_den, c):
    """k escalones con espaciados alternos g1, g2, g1, ... colocados simetricamente.
    Retraso por escalon h = (f_num/f_den) * P, con P = g1 + g2;  delta = -A*h."""
    gaps = [(g1, g2)[t % 2] for t in range(k - 1)]
    span = sum(gaps)
    if p - 2 - span < 0 or (p - 2 - span) % 2:
        return None
    r0 = (p - 2 - span) // 2
    h = (g1 + g2) * f_num * pow(f_den, p - 2, p) % p
    delta = (-A * h) % p
    if delta == 0:
        return None
    jumps, x = set(), r0
    for t in range(k):
        jumps.add(x)
        if t < k - 1:
            x += gaps[t]
    sol = [c % p]
    for step in range(p - 1):
        sol.append((sol[-1] + A + (delta if step in jumps else 0)) % p)
    return tuple(sol)


def family2(p, fracs=((1, 1), (1, 3))):
    out = {}
    for fn, fd in fracs:
        for A in range(1, p):
            for g1 in range(1, p - 1):
                for g2 in range(1, p - 1):
                    if g1 == g2 or g1 + g2 > p - 2:
                        continue
                    for k in range(3, p):
                        if (k // 2) * (g1 + g2) - g2 > p - 2:
                            break
                        for c in range(p):
                            s = build2(p, A, g1, g2, k, fn, fd, c)
                            if s is not None and s not in out and valid(s) and len({(s[i + 1] - s[i]) % p for i in range(p - 1)}) == 2:
                                out[s] = (fn, fd)
    return out
