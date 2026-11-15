"""
Ejemplos de las formulas generadoras de cada grupo p (nº de espirales de Prony)
para N = 7, 11, 13. Muestra las dos caras:
  - polinomio mod N:      sol[j] = (c0 + c1 j + ... ) mod N
  - descomposicion Prony: z_j = A_1 w_1^j + ... + A_p w_p^j   (suma de p espirales)
"""

import math
import numpy as np
from collections import defaultdict

from spiral_queens import SpiralQueens
from prony_sweep import solve_queens, gf_solve_square, complex_points


def poly_coeffs(sol, p):
    n = len(sol)
    V = [[pow(j, k, p) for k in range(n)] for j in range(n)]
    return gf_solve_square(V, [s % p for s in sol], p)


def fmt_poly(c, N):
    terms = []
    for k, a in enumerate(c):
        a %= N
        if a == 0:
            continue
        if k == 0:
            terms.append(f"{a}")
        elif k == 1:
            terms.append("j" if a == 1 else f"{a}j")
        else:
            terms.append(f"j^{k}" if a == 1 else f"{a}j^{k}")
    return f"({' + '.join(terms) if terms else '0'}) mod {N}"


def prony_decompose(z, tol=1e-6):
    N = len(z)
    scale = np.max(np.abs(z))
    for p in range(1, N // 2 + 1):
        if N - p < p:
            break
        A = np.array([[z[i - 1 - k] for k in range(p)] for i in range(p, N)])
        b = np.array([z[i] for i in range(p, N)])
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        if np.max(np.abs(A @ c - b)) / scale < tol:
            w = np.roots(np.concatenate(([1.0], -c)))
            V = np.array([[w[k] ** j for k in range(p)] for j in range(N)])
            amp, *_ = np.linalg.lstsq(V, z, rcond=None)
            return p, list(zip(amp, w))
    return N // 2 + 1, None


def fmt_modes(modes):
    out = []
    for A, w in modes:
        out.append(f"({abs(A):.2f}∠{math.degrees(np.angle(A)):+.0f}°)·"
                   f"[|w|={abs(w):.3f}, giro {math.degrees(np.angle(w)):+.1f}°/paso]")
    return out


def main():
    for N in (7, 11, 13):
        sq = SpiralQueens(N)
        sols = solve_queens(N)
        groups = defaultdict(list)
        for s in sols:
            p, modes = prony_decompose(complex_points(sq, s))
            if len(groups[p]) < 2:
                groups[p].append((s, modes))

        maxp = N // 2 + 1
        print("=" * 70)
        print(f"N = {N}   ({len(sols)} soluciones)")
        print("=" * 70)
        for p in sorted(groups):
            kind = ("espiral pura" if p == 1 else
                    "IRREDUCIBLE (nº max de espirales)" if p == maxp else
                    "intermedia")
            print(f"\n--- grupo p = {p}  ({kind}) ---")
            for s, modes in groups[p]:
                c = poly_coeffs(s, N)
                print(f"  sol = {s}")
                print(f"     polinomio : sol[j] = {fmt_poly(c, N)}")
                if p == 1:
                    A, w = modes[0]
                    m = round(np.angle(w) / (2 * math.pi / N)) % N
                    b = s[0]
                    print(f"     1 espiral : z_j = A·w^j   (m={m}, b={b};  "
                          f"|w|={abs(w):.3f}, giro {math.degrees(np.angle(w)):+.1f}°)")
                elif modes is not None:
                    print(f"     {p} espirales: z_j = " + " + ".join(f"A{k+1}·w{k+1}^j"
                          for k in range(p)))
                    for k, line in enumerate(fmt_modes(modes)):
                        print(f"        w{k+1}: {line}")
        print()


if __name__ == "__main__":
    main()
