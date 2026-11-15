"""
Barrido de grado (GF(N)) y modos de Prony sobre TODAS las soluciones de N reinas.

Pregunta: al crecer la complejidad, ¿el nº de parametros crece gradualmente
(aparecen p=2,3,4,5 intermedios) o el salto es binario (espiral o irreducible)?

Uso:  python prony_sweep.py 11
"""

import sys
import math
import numpy as np
from collections import Counter

from spiral_queens import SpiralQueens, is_prime


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


def gf_solve_square(V, b, p):
    """Resuelve V c = b (mod p) con V cuadrada invertible. Devuelve c."""
    n = len(b)
    M = [[V[i][k] % p for k in range(n)] + [b[i] % p] for i in range(n)]
    for col in range(n):
        piv = next(i for i in range(col, n) if M[i][col] % p != 0)
        M[col], M[piv] = M[piv], M[col]
        inv = pow(M[col][col], p - 2, p)
        M[col] = [(x * inv) % p for x in M[col]]
        for i in range(n):
            if i != col and M[i][col] % p != 0:
                f = M[i][col]
                M[i] = [(M[i][k] - f * M[col][k]) % p for k in range(n + 1)]
    return [M[i][n] % p for i in range(n)]


def poly_degree(sol, p):
    """Grado del polinomio unico mod p que interpola sol (Vandermonde nodos 0..p-1)."""
    n = len(sol)
    V = [[pow(j, k, p) for k in range(n)] for j in range(n)]
    c = gf_solve_square(V, [s % p for s in sol], p)
    d = max((k for k in range(n) if c[k] != 0), default=0)
    return d


def complex_points(sq, sol):
    theta = np.array([sq.a[v] for v in sol])
    return sq.r * np.exp(1j * theta)


def prony_min_p(z, tol=1e-6):
    N = len(z)
    scale = np.max(np.abs(z))
    maxp = N // 2
    for p in range(1, maxp + 1):
        if N - p < p:
            break
        A = np.array([[z[i - 1 - k] for k in range(p)] for i in range(p, N)])
        b = np.array([z[i] for i in range(p, N)])
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(A @ c - b)) / scale < tol:
            return p
    return maxp + 1


def main(N):
    sq = SpiralQueens(N)
    sols = solve_queens(N)
    print(f"N={N}: {len(sols)} soluciones\n")

    deg_p = []
    for s in sols:
        d = poly_degree(s, N)
        p = prony_min_p(complex_points(sq, s))
        deg_p.append((d, p))

    # distribucion de p
    pc = Counter(p for _, p in deg_p)
    print("Modos de Prony p:  (nº mínimo de espirales que generan la solución)")
    for p in sorted(pc):
        tag = " (espiral pura)" if p == 1 else (" (irreducible, ≥max)" if p == N // 2 + 1 else "")
        print(f"  p={p}: {pc[p]:4d} soluciones{tag}")

    # distribucion de grado
    dc = Counter(d for d, _ in deg_p)
    print("\nGrado del polinomio mod N:")
    for d in sorted(dc):
        print(f"  grado {d}: {dc[d]:4d}")

    # tabla cruzada grado x p
    print("\nTabla cruzada  (filas=grado, columnas=p):")
    ps = sorted(pc)
    print("  grado \\ p |" + "".join(f"{p:5d}" for p in ps))
    print("  ----------+" + "-----" * len(ps))
    for d in sorted(dc):
        row = Counter(pp for dd, pp in deg_p if dd == d)
        print(f"     {d:2d}     |" + "".join(f"{row.get(p,0):5d}" for p in ps))

    inter = sum(v for p, v in pc.items() if 2 <= p <= N // 2)
    print(f"\n¿Aparecen p intermedios (2..{N//2})?  {inter} soluciones"
          f"  ->  {'SÍ, complejidad gradual' if inter else 'NO, sigue binario'}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    if not is_prime(N):
        raise SystemExit(f"N={N} no es primo.")
    main(N)
