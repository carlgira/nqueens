"""
Nivel 1 limpio (sin las fusiones triviales que devuelven la propia toroidal) y descrito de una
forma que no depende de p, para poder comparar primos:

    pendiente a (con signo), y la FORMA del cambio: las filas que vienen de la segunda toroidal,
    medidas desde su centro (en medias filas). Ej. forma (-3,-1,1,3) = 4 filas alternas.

Clase de pendiente: {a, -a, 1/a, -1/a} -> se nombra por el menor |a|.

Uso: python nivel1_formas.py 11 13 17 19 23
"""
import sys
from collections import Counter, defaultdict
from nivel1 import level1, sg


def cls(a, p):
    inv = pow(a, -1, p)
    return min(abs(sg(x, p)) for x in (a, -a, inv, -inv))


def real(p):
    R = level1(p)
    out = []
    for t, (a, c, a2, c2, u, x, w, k) in R.items():
        if all((t[i + 1] - t[i]) % p == (t[1] - t[0]) % p for i in range(p - 1)):
            continue                                    # es la propia toroidal
        rows = [i for i in range(p) if t[i] != (a * i + c) % p]
        s = rows[0] + rows[-1]
        out.append((a, a2, tuple(2 * i - s for i in rows), len(rows), u))
    return out


if __name__ == '__main__':
    ps = [int(x) for x in sys.argv[1:]] or [11, 13, 17]
    data = {p: real(p) for p in ps}
    print("Soluciones de nivel 1 reales y relacion entre las dos pendientes:")
    for p in ps:
        D = data[p]
        opp = sum((a + a2) % p == 0 for a, a2, *_ in D)
        print(f"  p={p}: {len(D)}   opuestas (a, -a): {opp}   otras: {len(D)-opp}")
    print("\nPor clase de pendiente (filas = clase, columnas = p):")
    allc = sorted({cls(a, p) for p in ps for a, *_ in data[p]} | {2, 3, 4, 5})
    print("  clase | " + " ".join(f"{p:>6}" for p in ps))
    for c in allc:
        row = []
        for p in ps:
            n = sum(cls(a, p) == c for a, *_ in data[p])
            members = sorted({abs(sg(a, p)) for a in range(2, p - 1) if cls(a, p) == c})
            row.append(f"{n:>6}" if members else "     -")
        print(f"  {c:5d} | " + " ".join(row))
    print("\nPor n de filas cambiadas:")
    ks = sorted({d[3] for p in ps for d in data[p]})
    print("  filas | " + " ".join(f"{p:>6}" for p in ps))
    for k in ks:
        print(f"  {k:5d} | " + " ".join(f"{sum(d[3] == k for d in data[p]):>6}" for p in ps))
    for target in (2, 3):
        print(f"\nFORMAS con pendiente a = +{target} (cuantas soluciones da cada forma en cada p):")
        shapes = defaultdict(dict)
        for p in ps:
            for sh, n in Counter(d[2] for d in data[p] if sg(d[0], p) == target).items():
                shapes[sh][p] = n
        for sh in sorted(shapes, key=lambda s: (len(s), s)):
            print(f"  {str(sh):<34} " + " ".join(f"{shapes[sh].get(p, 0):>4}" for p in ps))
