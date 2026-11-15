"""
Regresion simbolica SIN biblioteca sobre las soluciones de las N reinas (N=7).

Sobre GF(7) toda funcion Z7->Z7 es un polinomio unico de grado <=6, y los
monomios {1, j, j^2, ..., j^6} forman la base COMPLETA del espacio de funciones
(no es una biblioteca elegida a mano). "Descubrir la formula" de una solucion
sol[j] = permutacion, es entonces hallar el polinomio de MENOR grado mod 7 que
la reproduce.  Ese grado minimo mide lo "bonita/simple" que es la solucion:

    grado 1  ->  sol[j] = (m*j + b) mod 7   (las espirales construibles)
    grado>=2 ->  soluciones no lineales ("feas")
"""

import random
from collections import Counter

P = 7


# ------------------------------------------------------ solver de N reinas
def solve_queens(n):
    sols = []

    def bt(col, rows, d1, d2, cur):
        if col == n:
            sols.append(cur[:]); return
        for r in range(n):
            if r in rows or (r - col) in d1 or (r + col) in d2:
                continue
            rows.add(r); d1.add(r - col); d2.add(r + col); cur.append(r)
            bt(col + 1, rows, d1, d2, cur)
            rows.remove(r); d1.remove(r - col); d2.remove(r + col); cur.pop()

    bt(0, set(), set(), set(), [])
    return sols


# ------------------------------------------------- algebra lineal sobre GF(p)
def gf_solve(A, b, p=P):
    """Resuelve A c = b (mod p). Devuelve (consistente, una solucion c)."""
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


def min_degree_poly(sol, p=P):
    """Menor grado d y coeficientes tal que sum c_k j^k == sol[j] (mod p)."""
    n = len(sol)
    for d in range(0, n):
        A = [[pow(j, k, p) for k in range(d + 1)] for j in range(n)]
        ok, c = gf_solve(A, [s % p for s in sol], p)
        if ok:
            return d, c
    return None, None


def fmt_poly(c, p=P):
    terms = []
    for k, coef in enumerate(c):
        coef %= p
        if coef == 0:
            continue
        if k == 0:
            terms.append(f"{coef}")
        elif k == 1:
            terms.append("j" if coef == 1 else f"{coef}j")
        else:
            terms.append(f"j^{k}" if coef == 1 else f"{coef}j^{k}")
    body = " + ".join(terms) if terms else "0"
    return f"({body}) mod {p}"


def verify(sol, c, p=P):
    return all(sum(ck * pow(j, k, p) for k, ck in enumerate(c)) % p == sol[j]
               for j in range(len(sol)))


def main():
    n = 7
    sols = solve_queens(n)
    print(f"N={n}: {len(sols)} soluciones totales\n")

    # --- primero: unas cuantas al azar ---
    random.seed(7)
    sample = random.sample(sols, 5)
    print("=== Formula descubierta para 5 soluciones al azar ===")
    for s in sample:
        d, c = min_degree_poly(s)
        okk = "OK" if verify(s, c) else "FALLA"
        print(f"  {s}   ->  sol[j] = {fmt_poly(c):28s} grado {d}  [{okk}]")

    # --- despues: las 40 completas ---
    print("\n=== Las 40 completas: clasificacion por grado ===")
    degs = []
    by_deg = {}
    for s in sols:
        d, c = min_degree_poly(s)
        assert verify(s, c), s
        degs.append(d)
        by_deg.setdefault(d, []).append((s, c))

    hist = Counter(degs)
    for d in sorted(hist):
        print(f"  grado {d}: {hist[d]:2d} soluciones")
    print(f"  ------------------------\n  total : {len(sols)}")

    print(f"\nLineales (grado 1, las 'espirales construibles'): {hist.get(1,0)}")
    print("Ejemplos de grado 1:")
    for s, c in by_deg.get(1, [])[:4]:
        print(f"    {s}  ->  {fmt_poly(c)}")
    hi = max(by_deg)
    print(f"\nEjemplos del grado maximo ({hi}, las 'feas'):")
    for s, c in by_deg.get(hi, [])[:4]:
        print(f"    {s}  ->  {fmt_poly(c)}")


if __name__ == "__main__":
    main()
