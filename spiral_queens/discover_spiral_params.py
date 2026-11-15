"""
Regresion simbolica sobre las 28 soluciones ESPIRALES de las 7 reinas.

Paso 1: para cada espiral, ajustar (regresion exacta sobre GF(7)) el polinomio
        de grado 1   sol[j] = (m*j + b) mod 7   -> parametros (m, b).
Paso 2: mirar si esos parametros (m, b) siguen a su vez una regla en funcion de
        las etiquetas de construccion (d = fraccion de phi, i = desfase angular,
        direccion = +-1). Es decir: regresion sobre los parametros mismos.
"""

from spiral_queens import SpiralQueens

P = 7


def gf_solve(A, b, p=P):
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


def fit_linear(sol, p=P):
    """Ajuste exacto sol[j] = (b + m*j) mod p. Devuelve (m, b) o None si no es lineal."""
    n = len(sol)
    A = [[pow(j, k, p) for k in range(2)] for j in range(n)]  # [1, j]
    ok, c = gf_solve(A, [s % p for s in sol], p)
    if not ok:
        return None
    b, m = c[0], c[1]
    if all((m * j + b) % p == sol[j] for j in range(n)):
        return m, b
    return None


def main():
    sq = SpiralQueens(7)
    labels = sq.init_sols_label()

    print("=== Paso 1: parametros (m, b) de cada espiral ===")
    print(" d  i  dir   sol                       -> sol[j] = (m j + b) mod 7")
    rows = []
    for sol, d, i, direction in labels:
        s = [int(x) for x in sol]
        mb = fit_linear(s)
        assert mb is not None, f"no lineal: {s}"
        m, b = mb
        rows.append((d, i, direction, m, b, s))
        tag = "s1" if direction == 1 else "s2"
        print(f" {d}  {i}  {tag}   {str(s):24s} -> m={m}, b={b}")

    # === Paso 2: regresion sobre los parametros ===
    # Hipotesis A:  m = (dir * 2^d) mod 7
    # Hipotesis B:  b = (m - i) mod 7
    print("\n=== Paso 2: ¿los parametros (m,b) siguen una regla? ===")
    okA = all(m == (direction * pow(2, d, P)) % P for d, i, direction, m, b, s in rows)
    okB = all(b == (m - i) % P for d, i, direction, m, b, s in rows)
    print(f"  Hipotesis A:  m = (dir · 2^d) mod 7      -> {'SE CUMPLE' if okA else 'no'} (28/28)"
          if okA else f"  Hipotesis A: no")
    print(f"  Hipotesis B:  b = (m - i) mod 7          -> {'SE CUMPLE' if okB else 'no'} (28/28)"
          if okB else f"  Hipotesis B: no")

    # Formula unificada:  sol[j] = (m*(j+1) - i) mod 7  con m = dir*2^d
    print("\n=== Formula unica para las 28 espirales ===")
    print("  m = (dir · 2^d) mod 7     b = (m - i) mod 7")
    print("  =>  sol[j] = (m·j + b) mod 7 = (m·(j+1) - i) mod 7\n")

    all_ok = True
    for d, i, direction, m, b, s in rows:
        mm = (direction * pow(2, d, P)) % P
        pred = [(mm * (j + 1) - i) % P for j in range(7)]
        if pred != s:
            all_ok = False
            print(f"  FALLA d={d} i={i} dir={direction}: pred={pred} real={s}")
    print(f"  Verificacion: la formula reproduce las 28 -> {'SI ✓' if all_ok else 'NO'}")

    # Resumen del mapa d -> m
    print("\n  Mapa d -> m (potencias de 2 mod 7):")
    for direction, tag in [(1, "s1"), (-1, "s2")]:
        ms = {d: (direction * pow(2, d, P)) % P for d in (1, 2)}
        print(f"    {tag}: " + ", ".join(f"d={d}->m={m}" for d, m in ms.items()))


if __name__ == "__main__":
    main()
