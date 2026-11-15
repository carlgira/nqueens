"""
¿Cuantas soluciones dan, para cada N, las reglas exactas encontradas?

Reglas (todas generan por formula; cada solucion se VALIDA antes de contarla):
  T  toroidal de un tramo:      sigma(i) = a*i + c,  mcd(a, N) = mcd(a±1, N) = 1
  G  dos mitades de paso s par: salto j = s^2+s-1 + 2m(s^2-1)   [mcd(N, s(s-1)(s+1)) = 2]
     y el caso (2, 1, 2)
  H  tramo largo de paso 2 con los dos extremos desplazados: alfas (4, 2, 4)   [N = 1 mod 6]
  V  sube y baja: ((N-1)/2, 2, -(N-1)/2) y ((N+1)/3, (N-5)/3, -(N+1)/3)        [N = 5 mod 6]
  3  tres tercios de paso 3, saltos (7,7), (11,11), (7,19), (19,7), (2,2)      [N multiplo de 3]
  P  g bloques de paso g (g = 5, 7), saltos iguales g + m*N/g, m = 2..g-2     [mcd(g, N/g) = 1]

Se cuentan soluciones distintas, y tambien tras cerrar bajo las 8 simetrias del tablero.

Uso: python reglas.py [--hasta 60]
"""
import argparse, math
from collections import OrderedDict
import min_waves as mw

Q = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712, 14: 365596,
     15: 2279184, 16: 14772512, 17: 95815104, 18: 666090624, 19: 4968057848, 20: 39029188884,
     21: 314666222712, 22: 2691008701644, 23: 24233937684440, 24: 227514171973736,
     25: 2207893435808352, 26: 22317699616364044, 27: 234907967154122528}


def valid(sol):
    N = len(sol)
    return len(set(sol)) == N and len({i + c for i, c in enumerate(sol)}) == N and len({i - c for i, c in enumerate(sol)}) == N


def from_runs(N, alfas, lens, c0):
    sol = [c0 % N]
    for a, l in zip(alfas, lens):
        for _ in range(l):
            sol.append((sol[-1] + a) % N)
    return tuple(sol)


def gen(N, alfas, lens, out):
    """todas las columnas iniciales, con espejo (alfas -> -alfas) y reverso"""
    assert sum(lens) == N - 1
    for al, ln in ((alfas, lens), (alfas[::-1], lens[::-1])):
        for sign in (1, -1):
            a2 = [sign * x for x in al]
            for c0 in range(N):
                s = from_runs(N, a2, ln, c0)
                if valid(s):
                    out.add(s)


def rules(N):
    R = OrderedDict((k, set()) for k in 'TGHV3P')
    # T
    for a in range(2, N - 1):
        if math.gcd(a, N) == 1 and math.gcd(a + 1, N) == 1 and math.gcd(a - 1, N) == 1:
            for c in range(N):
                R['T'].add(tuple((a * i + c) % N for i in range(N)))
    # G
    if N % 2 == 0:
        h = N // 2
        for s in range(2, N, 2):
            if math.gcd(N, s * (s - 1) * (s + 1)) != 2:
                continue
            m = 0
            while N - (s * s + s + 1) - 2 * m * (s + 1) ** 2 > 0:
                j = s * s + s - 1 + 2 * m * (s * s - 1)
                gen(N, [s, j, s], [h - 1, 1, h - 1], R['G'])
                m += 1
        gen(N, [2, 1, 2], [h - 1, 1, h - 1], R['G'])
    # H
    if N % 6 == 1 and N >= 7:
        gen(N, [4, 2, 4], [1, N - 3, 1], R['H'])
    # V
    if N % 6 == 5 and N >= 11:
        gen(N, [(N - 1) // 2, 2, -(N - 1) // 2], [3, 1, N - 5], R['V'])
        gen(N, [(N + 1) // 3, (N - 5) // 3, -(N + 1) // 3], [4, 1, N - 6], R['V'])
    # 3
    if N % 3 == 0 and N >= 6:
        t = N // 3
        for j1, j2 in ((7, 7), (11, 11), (7, 19), (19, 7), (2, 2)):
            gen(N, [3, j1, 3, j2, 3], [t - 1, 1, t - 1, 1, t - 1], R['3'])
    # P
    for g in (5, 7):
        if N % g == 0 and N // g > 1 and math.gcd(g, N // g) == 1:
            p = N // g
            for m in range(2, g - 1):
                j = g + m * p
                alf, ln = [], []
                for b in range(g):
                    alf.append(g); ln.append(p - 1)
                    if b < g - 1:
                        alf.append(j); ln.append(1)
                gen(N, alf, ln, R['P'])
    return R


def closure(S, N):
    G = mw.board_maps(N)
    out = set()
    for s in S:
        for g in G:
            out.add(mw.apply_map(g, s))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--desde', type=int, default=4)
    ap.add_argument('--hasta', type=int, default=60)
    a = ap.parse_args()
    print(f"{'N':>3} {'mod6':>4} | {'T':>6} {'G':>6} {'H':>5} {'V':>4} {'3':>5} {'P':>6} | {'distintas':>9} {'con simetrias':>13} | Q(N) y fraccion")
    for N in range(a.desde, a.hasta + 1):
        R = rules(N)
        S = set().union(*R.values())
        C = closure(S, N)
        frac = ''
        if N in Q:
            f = len(C) / Q[N]
            frac = f"{Q[N]}  {100*f:.1f}%" if f >= 0.001 else f"{Q[N]:.3g}  {f:.1e}"
        print(f"{N:3d} {N%6:4d} | " + " ".join(f"{len(R[k]) or '·':>{w}}" for k, w in zip('TGHV3P', (6, 6, 5, 4, 5, 6)))
              + f" | {len(S):9d} {len(C):13d} | {frac}", flush=True)
