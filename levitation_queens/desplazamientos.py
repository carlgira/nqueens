"""
N = p primo. Regla "intercambios a distancia fija":

    toroidal de un tramo  L(i) = A*i + c,  y un conjunto U de filas tal que cada fila i de U
    se intercambia con la fila i + d  (misma distancia d para todas; U y U+d disjuntos).

    sigma(i) = A*(i+d) + c   si i en U          (sube un escalon  delta = A*d)
    sigma(i) = A*(i-d) + c   si i-d en U        (baja un escalon)
    sigma(i) = A*i + c       en otro caso

Es la lectura "con retraso" de las soluciones de 3 alfas con escalones opuestos
(alfas A, A+delta, A-delta). Un solo par (|U| = 1) coincide con un reflejo.

Construccion: para cada (A, d, c) se decide fila a fila si se intercambia o no, colocando
las reinas y podando al primer ataque (no se enumeran las Q(p) soluciones).

Uso: python desplazamientos.py 11 13 17 19 23
"""
import json, os, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def build(p):
    out = {}
    cols = [False] * p
    d1 = [False] * (2 * p)
    d2 = [False] * (2 * p)
    sol = [0] * p

    def put(r, x):
        if cols[x] or d1[r + x] or d2[r - x + p]:
            return False
        cols[x] = d1[r + x] = d2[r - x + p] = True
        sol[r] = x
        return True

    def rem(r, x):
        cols[x] = d1[r + x] = d2[r - x + p] = False

    for A in range(2, p - 1):
        for d in range(1, p):
            for c in range(p):
                taken = [False] * p            # fila ya colocada como pareja de otra

                def rec(i, nsw):
                    while i < p and taken[i]:
                        i += 1
                    if i == p:
                        if nsw:
                            s = tuple(sol)
                            if s not in out or nsw < out[s][2]:
                                out[s] = (A, d, nsw)
                        return
                    x = (A * i + c) % p
                    if put(i, x):                         # fila sin tocar
                        rec(i + 1, nsw)
                        rem(i, x)
                    j = i + d
                    if j < p and not taken[j]:           # intercambiar i <-> i+d
                        xi, xj = (A * j + c) % p, x
                        if put(i, xi):
                            if put(j, xj):
                                taken[j] = True
                                rec(i + 1, nsw + 1)
                                taken[j] = False
                                rem(j, xj)
                            rem(i, xi)
                rec(0, 0)
    return out


def tor(s):
    N = len(s)
    return len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N


if __name__ == '__main__':
    print(f"{'p':>3} | soluciones | toroidales | no toroidales | pares intercambiados -> soluciones | seg")
    for p in [int(x) for x in sys.argv[1:]] or [7, 11, 13, 17, 19]:
        t0 = time.time()
        S = build(p)
        nt = sum(not tor(s) for s in S)
        k = Counter(v[2] for v in S.values())
        print(f"{p:3d} | {len(S):7d} | {len(S)-nt:6d} | {nt:7d} | {dict(sorted(k.items()))} | {time.time()-t0:.0f}", flush=True)
        json.dump([list(s) for s in S], open(os.path.join(HERE, 'data', f'desplazamientos_p{p}.json'), 'w'))


# ---------------------------------------------------------------- rotaciones de 3 filas
def build3(p, with_swaps=False):
    """Un tramo A*i + c donde ternas de filas (i, i+d, i+2d) rotan sus columnas (en uno u
    otro sentido), todas con la misma distancia d. Con with_swaps=True se permiten ademas
    intercambios i <-> i+d en la misma solucion."""
    out = {}
    cols = [False] * p
    d1 = [False] * (2 * p)
    d2 = [False] * (2 * p)
    sol = [0] * p

    def put(r, x):
        if cols[x] or d1[r + x] or d2[r - x + p]:
            return False
        cols[x] = d1[r + x] = d2[r - x + p] = True
        sol[r] = x
        return True

    def rem(r, x):
        cols[x] = d1[r + x] = d2[r - x + p] = False

    for A in range(2, p - 1):
        for d in range(1, (p - 1) // 2 + 1):
            for c in range(p):
                taken = [False] * p
                L = [(A * i + c) % p for i in range(p)]

                def place(rows, xs):
                    done = []
                    for r, x in zip(rows, xs):
                        if put(r, x):
                            done.append((r, x))
                        else:
                            for rr, xx in done:
                                rem(rr, xx)
                            return None
                    return done

                def rec(i, n3, n2):
                    while i < p and taken[i]:
                        i += 1
                    if i == p:
                        if n3:
                            s = tuple(sol)
                            if s not in out or (n3 + n2) < sum(out[s][2:]):
                                out[s] = (A, d, n3, n2)
                        return
                    if put(i, L[i]):
                        rec(i + 1, n3, n2)
                        rem(i, L[i])
                    j, k = i + d, i + 2 * d
                    if k < p and not taken[j] and not taken[k]:
                        for xs in ((L[j], L[k], L[i]), (L[k], L[i], L[j])):      # los dos sentidos
                            done = place((i, j, k), xs)
                            if done:
                                taken[j] = taken[k] = True
                                rec(i + 1, n3 + 1, n2)
                                taken[j] = taken[k] = False
                                for rr, xx in done:
                                    rem(rr, xx)
                    if with_swaps and j < p and not taken[j]:
                        done = place((i, j), (L[j], L[i]))
                        if done:
                            taken[j] = True
                            rec(i + 1, n3, n2 + 1)
                            taken[j] = False
                            for rr, xx in done:
                                rem(rr, xx)
                rec(0, 0, 0)
    return out
