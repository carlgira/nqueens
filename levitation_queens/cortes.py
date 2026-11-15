"""
N = p primo. Regla unificada por CORTES:

    una solucion con J cortes = una toroidal de un tramo (pendiente A) cuyas filas se cortan en
    J+1 bloques consecutivos que se recolocan en otro orden.

Equivalente: en la palabra de alfas, todos los pasos valen A salvo J (los cortes, de cualquier
valor). Cortes = (N-1) - (veces que aparece el alfa dominante).

Construccion completa por backtracking: para cada A y columna inicial, en cada fila se sigue
con el paso A o se gasta un corte (cualquier otro paso), podando al primer ataque. Genera
TODAS las soluciones con a lo sumo J cortes, sin enumerar las Q(N).

Uso: python cortes.py --J 4 --ns 7 11 13 17 19 23
Salida: data/cortes_J{J}.json   {N: [[solucion, A, cortes], ...]}
"""
import argparse, json, os, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def build(N, J):
    out = {}
    cols = [False] * N
    d1 = [False] * (2 * N)
    d2 = [False] * (2 * N)
    sol = [0] * N

    def rec(i, c, A, used):
        if i == N - 1:
            s = tuple(sol)
            if s not in out or used < out[s][1]:
                out[s] = (A, used)
            return
        r = i + 1
        left = N - 1 - i                     # pasos que quedan
        for step in ([A] if used == J else range(1, N)):
            cost = 0 if step == A else 1
            c2 = (c + step) % N
            if cols[c2] or d1[r + c2] or d2[r - c2 + N]:
                continue
            cols[c2] = d1[r + c2] = d2[r - c2 + N] = True
            sol[r] = c2
            rec(r, c2, A, used + cost)
            cols[c2] = d1[r + c2] = d2[r - c2 + N] = False

    for A in range(2, N - 1):
        for c0 in range(N):
            cols[c0] = d1[c0] = d2[N - c0] = True
            sol[0] = c0
            rec(0, c0, A, 0)
            cols[c0] = d1[c0] = d2[N - c0] = False
    return out


def tor(s):
    N = len(s)
    return len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--J', type=int, default=4)
    ap.add_argument('--ns', type=int, nargs='+', default=[7, 11, 13, 17, 19])
    ap.add_argument('--max-seg', type=float, default=600)
    a = ap.parse_args()
    path = os.path.join(HERE, 'data', f'cortes_J{a.J}.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    print(f"soluciones con a lo sumo {a.J} cortes")
    print(f"{'N':>3} | total | por n de cortes | toroidales | seg")
    for N in a.ns:
        t0 = time.time()
        S = build(N, a.J)
        res[str(N)] = [[list(s), A, j] for s, (A, j) in S.items()]
        c = Counter(j for _, j in S.values())
        dt = time.time() - t0
        print(f"{N:3d} | {len(S):6d} | {dict(sorted(c.items()))} | {sum(tor(s) for s in S)} | {dt:.1f}", flush=True)
        json.dump(res, open(path, 'w'))
        if dt > a.max_seg:
            break
