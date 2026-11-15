"""
¿La complejidad de Prony (p) se explica por la ESTRUCTURA del polinomio mod N?
   - paridad: ¿solo potencias impares / solo pares / mixto?
   - dispersion: nº de coeficientes no nulos
   - simetria de tablero: R180
Y dentro de cada grupo p, ¿hay sub-grupos por esas propiedades?

Uso: python subgroups.py 13
"""

import sys
import numpy as np
from collections import defaultdict, Counter

from spiral_queens import SpiralQueens
from prony_sweep import solve_queens, gf_solve_square, complex_points, prony_min_p


def poly_coeffs(sol, p):
    n = len(sol)
    V = [[pow(j, k, p) for k in range(n)] for j in range(n)]
    return gf_solve_square(V, [s % p for s in sol], p)


def parity_class(c):
    has_odd = any(c[k] for k in range(1, len(c), 2))
    has_even = any(c[k] for k in range(0, len(c), 2))
    if has_odd and not has_even:
        return "impar"
    if has_even and not has_odd:
        return "par"
    return "mixto"


def is_r180(sol, N):
    return all(sol[N - 1 - j] == N - 1 - sol[j] for j in range(N))


def main(N):
    sq = SpiralQueens(N)
    sols = solve_queens(N)

    by_p = defaultdict(list)
    for s in sols:
        p = prony_min_p(complex_points(sq, s))
        c = poly_coeffs(s, N)
        by_p[p].append((s, c))

    maxp = N // 2 + 1
    print(f"N={N}: {len(sols)} soluciones\n")
    print("  p | count | %impar %par %mixto | %R180 | dispersion(min–max, media)")
    print("  --+-------+--------------------+-------+---------------------------")
    for p in sorted(by_p):
        grp = by_p[p]
        n = len(grp)
        par = Counter(parity_class(c) for _, c in grp)
        r180 = sum(is_r180(s, N) for s, _ in grp)
        spars = [sum(1 for x in c if x) for _, c in grp]
        tag = "espiral" if p == 1 else ("IRRED" if p == maxp else "interm")
        print(f"  {p:2d}| {n:5d} | {100*par['impar']/n:5.0f} {100*par['par']/n:4.0f} "
              f"{100*par['mixto']/n:5.0f} | {100*r180/n:4.0f}% | "
              f"{min(spars):2d}–{max(spars):2d}, {np.mean(spars):4.1f}   [{tag}]")

    # sub-grupos dentro de cada p: desglose por (paridad, dispersion)
    print("\nSub-grupos dentro de cada p (paridad × dispersión):")
    for p in sorted(by_p):
        grp = by_p[p]
        sub = Counter((parity_class(c), sum(1 for x in c if x)) for _, c in grp)
        if len(sub) == 1:
            (par, sp), cnt = next(iter(sub.items()))
            print(f"  p={p}: UNICO sub-grupo -> {par}, dispersión {sp}  ({cnt})")
        else:
            desc = ", ".join(f"{par}/disp{sp}:{cnt}" for (par, sp), cnt in sorted(sub.items(), key=lambda e: -e[1])[:6])
            print(f"  p={p}: {len(sub)} sub-grupos -> {desc}{' ...' if len(sub)>6 else ''}")


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    main(N)
