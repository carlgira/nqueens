"""
Catalogo de soluciones por tramos (N pequenos) para inducir reglas.

Lee data/tramos_R{R}.json (python tramos.py --R R ...). Cada solucion tiene una firma:
    alfas     = paso de cada tramo, con signo, en (-N/2, N/2]
    longitudes = n de pasos de cada tramo
Forma canonica: la menor entre la firma, su espejo (alfas -> -alfas) y su reverso
(orden invertido), para no contar dos veces lo mismo.

Tipos (por la forma, sin mirar los valores):
    un tramo "largo" (L) tiene >= 2 pasos; un "salto" (j) tiene 1 paso.
    p.ej.  L j L  con alfas (2, 5, 2):  dos mitades de paso 2 unidas por un salto de 5.

Salidas:
  1. por N: cuantas soluciones de cada forma (L j L, L L L, j L j, ...) y cuantos alfas distintos
  2. tabla patron de alfas x N (los que aparecen en varios N)

Uso: python catalogo.py --R 3 [--min-n 3]
"""
import argparse, json, os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def sg(x, N):
    x %= N
    return x if x <= N // 2 else x - N


def canon(steps, lens, N):
    a = tuple(sg(s, N) for s in steps)
    cands = []
    for sign in (1, -1):
        for rev in (False, True):
            st = tuple(sg(sign * s, N) for s in a)
            ln = tuple(lens)
            if rev:
                st, ln = st[::-1], ln[::-1]
            cands.append((st, ln))
    # preferir la forma con primer alfa positivo y menor en valor absoluto
    return min(cands, key=lambda c: (tuple(abs(x) for x in c[0]), tuple(-x for x in c[0]), c[1]))


def shape(lens):
    return ' '.join('L' if l >= 2 else 'j' for l in lens)


def isprime(n):
    return n > 1 and all(n % p for p in range(2, int(n ** .5) + 1))


def main(R, min_n):
    d = json.load(open(os.path.join(HERE, 'data', f'tramos_R{R}.json')))
    Ns = sorted(int(n) for n in d)
    byN_shape = defaultdict(Counter)
    byN_alfas = defaultdict(Counter)
    pat = defaultdict(Counter)            # patron de alfas -> N -> cuantas
    patlen = defaultdict(lambda: defaultdict(set))
    for N in Ns:
        for sol, st, ln in d[str(N)]:
            if len(st) == 1:
                byN_shape[N]['1 tramo'] += 1
                byN_alfas[N][1] += 1
                continue
            cs, cl = canon(st, ln, N)
            byN_shape[N][shape(cl)] += 1
            byN_alfas[N][len(set(cs))] += 1
            pat[(shape(cl), cs)][N] += 1
            patlen[(shape(cl), cs)][N].add(cl)

    shapes = sorted({s for N in Ns for s in byN_shape[N]}, key=lambda s: (len(s), s))
    print(f"1. FORMAS (hasta {R} tramos).  L = tramo largo (>= 2 pasos), j = salto (1 paso)")
    print(f"{'N':>3} {'mod6':>4} {'primo':>5} | " + " | ".join(f"{s:>9}" for s in shapes) + " | alfas distintos {n: soluciones}")
    for N in Ns:
        print(f"{N:3d} {N%6:4d} {'si' if isprime(N) else '':>5} | "
              + " | ".join(f"{byN_shape[N].get(s, 0) or '·':>9}" for s in shapes)
              + f" | {dict(sorted(byN_alfas[N].items()))}")

    print(f"\n2. PATRONES DE ALFAS que aparecen en al menos {min_n} valores de N  (casilla = n de soluciones)")
    keep = sorted((k for k, v in pat.items() if len(v) >= min_n), key=lambda k: (len(k[1]), tuple(abs(x) for x in k[1]), k))
    print(f"{'forma':>7} {'alfas':>14} | " + " ".join(f"{N:>3}" for N in Ns))
    for k in keep:
        print(f"{k[0]:>7} {str(k[1]):>14} | " + " ".join(f"{pat[k].get(N, 0) or '·':>3}" for N in Ns))
    json.dump({f'{k[0]}|{k[1]}': {str(N): [v, sorted(map(list, patlen[k][N]))] for N, v in c.items()} for k, c in pat.items()},
              open(os.path.join(HERE, 'data', f'catalogo_R{R}.json'), 'w'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, default=3)
    ap.add_argument('--min-n', type=int, default=3)
    a = ap.parse_args()
    main(a.R, a.min_n)
