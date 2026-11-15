"""
Formula para las parejas de velocidades que funcionan con CUALQUIER giro (mezclas toroidales).

Espirales A, B con pendientes alfa = s1*2*d1, beta = s2*2*d2 (mod p).  Sea H el subgrupo de Z_p*
generado por
        m = beta/alfa,    u = (alfa+1)/(beta+1),    w = (alfa-1)/(beta-1).
Prediccion:  la pareja (d1,s1) x (d2,s2) funciona para todos los giros  <=>  H != Z_p*.
Soluciones toroidales por cada pareja de giros: 2^k - 2, k = (p-1)/|H| (filas elegidas = union de
clases de H, contadas desde la fila de cruce).

Ademas: soluciones NO toroidales segun el orden de m (para saber que tipos hay que probar a mano).

Uso: python formula_espirales.py 11 13 17 19 23 29 31
"""
import sys
from collections import defaultdict, Counter
from espirales import spiral, mix
from espirales_transfer import useful, labels


def subgroup(gens, p):
    H = {1}; frontier = [1]
    while frontier:
        x = frontier.pop()
        for g in gens:
            y = x * g % p
            if y not in H:
                H.add(y); frontier.append(y)
    return H


def order(m, p):
    k, x = 1, m % p
    while x != 1:
        x = x * m % p; k += 1
    return k


def predict(p):
    types = {}
    vs = [(d, s) for d in range(1, p // 2) for s in (1, -1)]
    for i, (d1, s1) in enumerate(vs):
        for (d2, s2) in vs[i + 1:]:
            al, be = s1 * 2 * d1 % p, s2 * 2 * d2 % p
            if al == be:
                continue
            inv = lambda x: pow(x, -1, p)
            gens = [be * inv(al) % p, (al + 1) * inv(be + 1) % p, (al - 1) * inv(be - 1) % p]
            H = subgroup(gens, p)
            types[((d1, s1), (d2, s2))] = (len(H) < p - 1, (p - 1) // len(H), order(gens[0], p))
    return types


def toro(t, p):
    return len({(r + c) % p for r, c in enumerate(t)}) == p and len({(r - c) % p for r, c in enumerate(t)}) == p


if __name__ == '__main__':
    for p in [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]:
        T = predict(p)
        pred = {k for k, v in T.items() if v[0]}
        print(f"===== p={p}: formula -> {len(pred)} parejas de velocidades 'siempre': " +
              ", ".join(f"v{a[0]}{'+' if a[1] > 0 else '-'}/v{b[0]}{'+' if b[1] > 0 else '-'}(k={T[(a, b)][1]})" for a, b in sorted(pred)))
        npred_tor = sum(p * p * (2 ** T[k][1] - 2) for k in pred)
        print(f"   toroidales de 2 rectas predichas: {npred_tor}")
        if p > 23:
            continue
        U = useful(p); L = labels(p)
        cnt = defaultdict(lambda: [0, 0])
        tor = 0; nontor_by_ord = Counter(); nontor_types = Counter()
        for x in range(len(L)):
            for y in range(x + 1, len(L)):
                a, b = sorted([L[x][:2], L[y][:2]], key=lambda v: (v[0], -v[1]))
                k = (a, b) if (a, b) in T else (b, a)
                if k not in T:
                    continue
                cnt[k][0] += 1
                if (L[x], L[y]) in U:
                    cnt[k][1] += 1
                    for t in mix(spiral(*L[x], p), spiral(*L[y], p), p)[1]:
                        if toro(t, p):
                            tor += 1
                        else:
                            nontor_by_ord[T[k][2]] += 1; nontor_types[k] += 1
        real = {k for k, (n, u) in cnt.items() if n == u}
        print(f"   real: {len(real)} 'siempre'; coinciden con la formula: {real == pred}"
              f" (falsos +: {len(pred - real)}, falsos -: {len(real - pred)}); toroidales reales: {tor}")
        print(f"   no toroidales segun el orden de m: {dict(sorted(nontor_by_ord.items()))}; tipos implicados: {len(nontor_types)} de {len(T)}")
