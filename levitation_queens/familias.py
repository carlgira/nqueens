"""
Familia cerrada para N = 2, 4 (mod 6): dos mitades de paso +2 unidas por un salto j.

    sigma(i) = c0 + 2i                      i < N/2
    sigma(i) = sigma(N/2 - 1) + j + 2(i - N/2)   i >= N/2        (todo mod N)

Reglas medidas (comprobadas hasta N = 1000):
  - j = 5 + 6m  (m = 0, 1, 2, ...) da exactamente  N - 7 - 18m  posiciones iniciales c0
  - j = 1 da exactamente 1 solucion cuando N = 0 o 4 (mod 6)
Ondas (j = 5):  k1 = (1, N/2),  k2 = (N - 2v, v) con v = impar mas cercano a N/3.

Uso: python familias.py 10 14 16 20 26 40 100
"""
import sys
from transfer import lp_margin


def valid(sol):
    N = len(sol)
    return len(set(sol)) == N and len({i + c for i, c in enumerate(sol)}) == N and len({i - c for i, c in enumerate(sol)}) == N


def fam(N, j, c0, s=2):
    h = N // 2
    sol = [(c0 + s * i) % N for i in range(h)]
    b = (sol[-1] + j) % N
    return sol + [(b + s * i) % N for i in range(h)]


def waves(N):
    v = round(N / 3)
    if v % 2 == 0:
        v = v + 1 if abs(v + 1 - N / 3) < abs(v - 1 - N / 3) else v - 1
    return [(1, N // 2), (N - 2 * v, v)]


if __name__ == '__main__':
    for N in [int(a) for a in sys.argv[1:]] or [10, 14, 16, 20, 26, 40]:
        counts = {j: sum(valid(fam(N, j, c)) for c in range(N)) for j in range(1, N) if j != 2}
        counts = {j: v for j, v in counts.items() if v}
        pred = {5 + 6 * m: N - 7 - 18 * m for m in range(N) if N - 7 - 18 * m > 0} if N % 6 in (2, 4) else {}
        if N % 6 in (0, 4):
            pred[1] = 1
        ks = waves(N)
        c0s = [c for c in range(N) if valid(fam(N, 5, c))]
        ms = [lp_margin(tuple(fam(N, 5, c)), N, ks)[0] for c in c0s] if N <= 120 else []
        print(f"N={N} (N%6={N%6}): conteos por salto {counts}  | regla acierta: {counts == pred}"
              + (f" | ondas {ks}: {sum(m > 0 for m in ms)}/{len(ms)} levitadas, margen {min(ms):.4f}-{max(ms):.4f}" if ms else ''))
