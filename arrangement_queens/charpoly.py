"""Polinomio caracteristico chi_N(q) de la disposicion de reinas por el metodo de
cuerpos finitos (Athanasiadis): chi(q) = #{x in Z_q^N fuera de los hiperplanos}.

Reduccion buena: q primo mayor que la maxima suma de ganancias de un ciclo
(~N^2/2), porque la matriz de coeficientes es grafica (unimodular).  Se usan N
primos para interpolar chi(q)/q (grado N-1) y 2 primos extra para verificar.
Zaslavsky:  regiones = |chi(-1)|,  regiones acotadas (modulo traslacion) = |chi(1)|.
"""
import json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction as F
from itertools import permutations

def isprime(n): return n > 1 and all(n % d for d in range(2, int(n**.5) + 1))

def max_cycle_gain(N):
    # cota: ciclo hamiltoniano zig-zag 0,N-1,1,N-2,... ; suma de |i-j|
    order, lo, hi = [], 0, N - 1
    while lo <= hi:
        order.append(lo); lo += 1
        if lo <= hi: order.append(hi); hi -= 1
    return sum(abs(order[k] - order[(k + 1) % N]) for k in range(N))

def count(N, q):
    return int(subprocess.check_output(['./torus_count', str(N), str(q)]))

def solve_coeffs(pts):
    n = len(pts); A = [[F(x)**k for k in range(n)] + [F(y)] for x, y in pts]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0); A[c], A[p] = A[p], A[c]
        A[c] = [v / A[c][c] for v in A[c]]
        for r in range(n):
            if r != c and A[r][c]: f = A[r][c]; A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[k][n] for k in range(n)]

def ev(co, x): return sum(c * x**k for k, c in enumerate(co))

def Q(N):
    return sum(all(abs(p[i] - p[j]) != j - i for i in range(N) for j in range(i + 1, N))
               for p in permutations(range(N)))

def run(N):
    ps, p = [], max_cycle_gain(N)
    while len(ps) < N + 2:
        p += 1
        if isprime(p): ps.append(p)
    with ThreadPoolExecutor(12) as ex:
        vals = list(ex.map(lambda q: count(N, q), ps))
    pts = list(zip(ps, vals))
    co = solve_coeffs(pts[:N])            # chi(q)/q, grado N-1
    assert all(den == 1 for den in (c.denominator for c in co))
    ok = all(ev(co, q) == v for q, v in pts[N:])
    chi = [0] + [int(c) for c in co]       # chi(q) = q * (chi/q)
    res = dict(N=N, chi=chi, verified=ok, primes=ps,
               regions=abs(ev(chi, -1)), bounded=abs(ev(chi, 1)), Q=Q(N))
    print(json.dumps(res), flush=True)
    return res

if __name__ == '__main__':
    Ns = range(2, int(sys.argv[1]) + 1) if len(sys.argv) > 1 else range(2, 9)
    out = [run(N) for N in Ns]
    json.dump(out, open('charpoly.json', 'w'), indent=1)
