"""
Generaliza el descubrimiento de parametros de las espirales a cualquier primo N.

Para cada espiral ajusta  sol[j] = (m*j + b) mod N  (regresion exacta en GF(N)),
extrae (m,b) y sus etiquetas (d, i, direccion), y trata de descubrir la regla:
    - m(d) en funcion de d  (¿geometrica m=g^d?  ¿aditiva m=g0+step*d?)
    - b(i) en funcion de i.

Uso:  python discover_spiral_params_general.py 11
      python discover_spiral_params_general.py 13
"""

import sys
from spiral_queens import SpiralQueens, is_prime


def gf_solve(A, b, p):
    m, ncol = len(A), len(A[0])
    rows = [[A[i][k] % p for k in range(ncol)] + [b[i] % p] for i in range(m)]
    pivots, r = [], 0
    for c in range(ncol):
        piv = next((i for i in range(r, m) if rows[i][c] % p != 0), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        inv = pow(rows[r][c], p - 2, p)
        rows[r] = [(x * inv) % p for x in rows[r]]
        for i in range(m):
            if i != r and rows[i][c] % p != 0:
                f = rows[i][c]
                rows[i] = [(rows[i][k] - f * rows[r][k]) % p for k in range(ncol + 1)]
        pivots.append(c); r += 1
        if r == m:
            break
    for i in range(m):
        if all(rows[i][c] % p == 0 for c in range(ncol)) and rows[i][ncol] % p != 0:
            return False, None
    c = [0] * ncol
    for idx, col in enumerate(pivots):
        c[col] = rows[idx][ncol] % p
    return True, c


def fit_linear(sol, p):
    n = len(sol)
    A = [[pow(j, k, p) for k in range(2)] for j in range(n)]
    ok, c = gf_solve(A, [s % p for s in sol], p)
    if not ok:
        return None
    b, m = c[0], c[1]
    if all((m * j + b) % p == sol[j] for j in range(n)):
        return m, b
    return None


def order_mult(g, p):
    x, o = g % p, 1
    while x != 1:
        x = (x * g) % p
        o += 1
    return o


def main(N):
    sq = SpiralQueens(N)
    labels = sq.init_sols_label()
    print(f"===== N = {N} =====")
    print(f"espirales generadas: {len(labels)}   (esperado (N-3)*N = {(N-3)*N})\n")

    rows = []
    non_linear = 0
    bad_len = 0
    for sol, d, i, direction in labels:
        s = [int(x) for x in sol]
        if len(s) != N:
            bad_len += 1
            continue
        mb = fit_linear(s, N)
        if mb is None:
            non_linear += 1
            continue
        m, b = mb
        rows.append((d, i, direction, m, b))

    if bad_len or non_linear:
        print(f"  (aviso: {bad_len} soluciones incompletas, {non_linear} no lineales)\n")

    # b = (m - i) mod N ?
    okB = all(b == (m - i) % N for d, i, direction, m, b in rows)
    print(f"Regla de b:   b = (m - i) mod {N}   -> {'SE CUMPLE' if okB else 'NO'} "
          f"({sum(1 for d,i,dr,m,b in rows if b==(m-i)%N)}/{len(rows)})")

    # mapa d -> m para s1 (direction=+1) y s2 (-1)
    def dmap(dr):
        mp = {}
        consistent = True
        for d, i, direction, m, b in rows:
            if direction != dr:
                continue
            if d in mp and mp[d] != m:
                consistent = False
            mp[d] = m
        return mp, consistent

    m_s1, c1 = dmap(1)
    m_s2, c2 = dmap(-1)
    ds = sorted(m_s1)
    print(f"\nMapa d -> m:")
    print(f"  s1 (+): " + ", ".join(f"d={d}->{m_s1[d]}" for d in ds) +
          ("" if c1 else "   (INCONSISTENTE)"))
    print(f"  s2 (-): " + ", ".join(f"d={d}->{m_s2[d]}" for d in ds) +
          ("" if c2 else "   (INCONSISTENTE)"))

    # s2 = -s1 ?
    ok_neg = all(m_s2[d] == (-m_s1[d]) % N for d in ds)
    print(f"  ¿s2 = -s1 mod {N}?  {'SI' if ok_neg else 'NO'}")

    # ¿m(d) geometrica?  base g = m(d=1); test m(d) == g^d
    g = m_s1[ds[0]]
    ok_geo = all(m_s1[d] == pow(g, d, N) for d in ds)
    print(f"\n¿Regla geometrica  m = {g}^d mod {N}?  -> {'SI ✓' if ok_geo else 'no'}")
    if ok_geo:
        print(f"   base g = {g},  orden multiplicativo de {g} mod {N} = {order_mult(g, N)}")

    # ¿m(d) aditiva?  m(d) = g0 + step*d
    if len(ds) >= 2:
        step = (m_s1[ds[1]] - m_s1[ds[0]]) % N
        g0 = (m_s1[ds[0]] - step) % N
        ok_add = all(m_s1[d] == (g0 + step * d) % N for d in ds)
        print(f"¿Regla aditiva     m = ({g0} + {step}·d) mod {N}?  -> {'SI' if ok_add else 'no'}")

    # verificacion de formula final si es geometrica
    if ok_geo and okB:
        allok = True
        for d, i, direction, m, b in rows:
            pred = [((direction * pow(g, d, N)) * (j + 1) - i) % N for j in range(N)]
            real = [(m * j + b) % N for j in range(N)]
            if pred != real:
                allok = False
        print(f"\nFormula unica  sol[j] = (dir·{g}^d·(j+1) - i) mod {N}"
              f"  -> reproduce las {len(rows)}: {'SI ✓' if allok else 'NO'}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    if not is_prime(N):
        raise SystemExit(f"N={N} no es primo.")
    main(N)
