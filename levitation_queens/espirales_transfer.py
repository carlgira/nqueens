"""
¿Una pareja de espirales que funciona en p sigue funcionando en otro primo (o con giros multiplicados)?
Pareja util = mezclarlas da al menos una solucion (espirales.mix).

Descripciones de una pareja, para trasladarla de p a p':
  literal : (d1, s1, giro1, d2, s2, giro2) igual
  cruce   : velocidades y sentidos iguales + la casilla donde se cruzan (fila, columna) igual
  cruce±  : igual, pero fila y columna medidas desde el borde mas cercano (con signo)
Se compara con la tasa base: % de parejas utiles del mismo tipo (mismas velocidades/sentidos) en p'.

Uso: python espirales_transfer.py
"""
import json, os
from collections import defaultdict
from espirales import spiral, mix

HERE = os.path.dirname(os.path.abspath(__file__))
PS = [11, 13, 17, 19, 23]


def labels(p):
    return [(d, s, i) for d in range(1, p // 2) for i in range(p) for s in (1, -1)]


def useful(p):
    f = os.path.join(HERE, 'data', f'espirales_utiles_p{p}.json')
    if os.path.exists(f):
        return {tuple(map(tuple, x)) for x in json.load(open(f))}
    L = labels(p); U = []
    for x in range(len(L)):
        for y in range(x + 1, len(L)):
            if mix(spiral(*L[x], p), spiral(*L[y], p), p)[1]:
                U.append((L[x], L[y]))
    json.dump(U, open(f, 'w'))
    return set(U)


def cross(l1, l2, p):
    A, B = spiral(*l1, p), spiral(*l2, p)
    r = [r for r in range(p) if A[r] == B[r]]
    return (r[0], A[r[0]]) if r else None


def sgn(x, p):
    return x if x <= p // 2 else x - p


def key(l1, l2, p, mode):
    t = (l1[0], l1[1], l2[0], l2[1])
    if mode == 'literal':
        return t + (l1[2], l2[2])
    c = cross(l1, l2, p)
    if c is None:
        return None
    return t + (c if mode == 'cruce' else (sgn(c[0], p), sgn(c[1], p)))


if __name__ == '__main__':
    U = {p: useful(p) for p in PS}
    for p in PS:
        n = len(labels(p)); print(f"p={p}: {len(U[p])} parejas utiles de {n*(n-1)//2}")
    print("\nTraslado p -> p': de las parejas utiles en p (con velocidades validas en p'), % que siguen utiles en p'")
    print("   (entre parentesis: tasa base en p' para los mismos tipos de velocidad/sentido)")
    for p, q in zip(PS, PS[1:]):
        row = []
        for mode in ('literal', 'cruce', 'cruce±'):
            # indice de p' por clave
            idx = defaultdict(lambda: [0, 0])
            Lq = labels(q)
            for x in range(len(Lq)):
                for y in range(x + 1, len(Lq)):
                    k = key(Lq[x], Lq[y], q, mode)
                    if k is not None:
                        idx[k][0] += 1; idx[k][1] += (Lq[x], Lq[y]) in U[q]
            hit = tot = 0; base_n = base_u = 0
            typ = defaultdict(lambda: [0, 0])
            for k, (n_, u_) in idx.items():
                typ[k[:4]][0] += n_; typ[k[:4]][1] += u_
            for l1, l2 in U[p]:
                k = key(l1, l2, p, mode)
                if k is None or k not in idx:
                    continue
                n_, u_ = idx[k]
                tot += 1; hit += u_ > 0
                tn, tu = typ[k[:4]]; base_n += 1; base_u += tu / tn
            row.append(f"{mode}: {100*hit/max(tot,1):5.1f}% de {tot} (base {100*base_u/max(base_n,1):4.1f}%)")
        print(f"  {p} -> {q}:  " + "  |  ".join(row))
    print("\nMultiplicar los giros por k dentro del mismo p (velocidades iguales): % de parejas utiles que siguen utiles")
    for p in PS:
        out = []
        for k in (2, 3, p - 1):
            hit = tot = 0
            for l1, l2 in U[p]:
                m1 = (l1[0], l1[1], l1[2] * k % p); m2 = (l2[0], l2[1], l2[2] * k % p)
                pair = (m1, m2) if (m1, m2) in U[p] or labels(p).index(m1) < labels(p).index(m2) else (m2, m1)
                tot += 1; hit += (m1, m2) in U[p] or (m2, m1) in U[p]
            out.append(f"k={k}: {100*hit/tot:.1f}%")
        n = len(labels(p))
        print(f"  p={p}: " + "  ".join(out) + f"   (base {100*len(U[p])/(n*(n-1)/2):.1f}%)")
