"""
Fusion a ciegas con MEDIDOR de progreso basado en las formulas de niveles.

P_k(N) (polinomio exacto) = n de mitades posibles de k filas consecutivas = techo del indice.
medidor_k = (mitades distintas de las k primeras filas vistas en lo generado) / P_k(N)
Cuando el medidor deja de crecer, esa plantilla esta agotada.

Uso: python medidor.py 17 --niveles 5 --muestra 3000
"""
import argparse, random, time
import min_waves as mw
from fusion import wave_gates
from fusion_ciega import seed_rules, close_sym, fuse_halves

POLY = {1: [0, 1], 2: [2, -3, 1], 3: [-36, 30, -9, 1], 4: [840, -534, 139, -18, 1],
        5: [-24332, 13104, -3098, 407, -30, 1],
        6: [870920, -418390, 91480, -11755, 943, -45, 1],
        7: [-36693996, 16209916, -3336014, 417178, -34411, 1879, -63, 1]}
DESDE = {1: 1, 2: 2, 3: 3, 4: 7, 5: 11, 6: 17, 7: 23}


def Pk(N, k):
    return sum(c * N ** j for j, c in enumerate(POLY[k]))


def meter(have, N, ks):
    return {k: len({s[:k] for s in have}) / Pk(N, k) for k in ks}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--niveles', type=int, default=4)
    ap.add_argument('--muestra', type=int, default=3000)
    a = ap.parse_args()
    N = a.N
    ks = [k for k in (3, 4, 5, 6, 7) if N >= DESDE[k]]
    print(f"N={N}: techos por formula  " + "  ".join(f"P_{k} = {Pk(N, k)}" for k in ks))
    have = close_sym(seed_rules(N), N)
    gates = list(wave_gates(N))
    rng = random.Random(0)
    print(f"{'nivel':>5} | {'soluciones':>10} | " + " | ".join(f"medidor k={k}" for k in ks) + " | seg")
    t0 = time.time()
    for lev in range(a.niveles + 1):
        m = meter(have, N, ks)
        print(f"{lev:5d} | {len(have):10d} | " + " | ".join(f"{100*m[k]:10.2f}%" for k in ks) + f" | {time.time()-t0:.0f}", flush=True)
        if lev == a.niveles:
            break
        pieces = have if len(have) <= a.muestra else set(rng.sample(sorted(have), a.muestra))
        new, _, _ = fuse_halves(pieces, N, gates)
        new = close_sym(new, N) - have
        if not new:
            break
        have |= new
