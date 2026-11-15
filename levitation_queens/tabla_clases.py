"""
Tabla por primo: clase 1 (lineales) y clase 2 (dos espirales) separada en toroidales / no toroidales.

  toroidales de clase 2  : formula (formula_espirales.predict), sin probar nada
  no toroidales de clase 2: se prueban combinaciones (dos opciones por fila, con poda) en TODAS las
                            parejas de espirales, para comprobar tambien la formula y la regla
                            "solo ord(m) en {2,3,4}"

Uso: python tabla_clases.py 11 13 17 19 23 29 31
"""
import sys, time, json, os
from collections import Counter
from formula_espirales import predict, order
from espirales import spiral

HERE = os.path.dirname(os.path.abspath(__file__))
TORO_TOTAL = {11: 88, 13: 4524, 17: 140692, 19: 820496, 23: 128850048}   # OEIS A007705


def pair_solutions(A, B, p):
    out = []
    st = [(0, 0, 0, 0, 0, ())]
    while st:
        i, cols, d1, d2, used, t = st.pop()
        if i == p:
            if used == 3:
                out.append(t)
            continue
        opts = ((0, A[i]),) if A[i] == B[i] else ((1, A[i]), (2, B[i]))
        for k, v in opts:
            b = 1 << v; e = 1 << (i + v); g = 1 << (i - v + p)
            if cols & b or d1 & e or d2 & g:
                continue
            st.append((i + 1, cols | b, d1 | e, d2 | g, used | k, t + (v,)))
    return out


def toro(t, p):
    return len({(r + c) % p for r, c in enumerate(t)}) == p and len({(r - c) % p for r, c in enumerate(t)}) == p


if __name__ == '__main__':
    rows = []
    for p in [int(x) for x in sys.argv[1:]] or [11, 13, 17, 19, 23]:
        t0 = time.time()
        T = predict(p)
        formula = sum(p * p * (2 ** v[1] - 2) for v in T.values() if v[0])
        S = [((d, s, i), spiral(d, s, i, p)) for d in range(1, p // 2) for i in range(p) for s in (1, -1)]
        tor, ntor = set(), set(); by_ord = Counter()
        for x in range(len(S)):
            for y in range(x + 1, len(S)):
                (la, A), (lb, B) = S[x], S[y]
                m = ((B[1] - B[0]) * pow(A[1] - A[0], -1, p)) % p
                for t in pair_solutions(A, B, p):
                    if toro(t, p):
                        tor.add(t)
                    else:
                        ntor.add(t); by_ord[order(m, p)] += 1
        r = dict(p=p, lineales=p * (p - 3), tor2=len(tor), tor2_formula=formula, notor2=len(ntor),
                 ord=dict(sorted(by_ord.items())), seg=round(time.time() - t0))
        rows.append(r)
        print(r, flush=True)
        json.dump(rows, open(os.path.join(HERE, 'data', 'tabla_clases.json'), 'w'))
    print(f"\n{'p':>3} | {'clase 1 (lineales, toroidales)':>30} | {'clase 2 toroidales':>18} | {'clase 2 no toroidales':>21} | toroidales totales (A007705)")
    for r in rows:
        print(f"{r['p']:3d} | {r['lineales']:30d} | {r['tor2']:18d} | {r['notor2']:21d} | {TORO_TOTAL.get(r['p'], '?')}")
