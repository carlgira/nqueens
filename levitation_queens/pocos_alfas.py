"""
Construir directamente las soluciones con POCOS ALFAS DISTINTOS (no pocos tramos).

alfa = sigma(i+1) - sigma(i) mod N. Se buscan todas las soluciones cuyos N-1 pasos usan a lo
sumo D valores distintos (D = 2 o 3), por backtracking fila a fila: en cada fila solo se
puede repetir un alfa ya usado o estrenar uno nuevo si quedan.

Para cada solucion: conjunto de alfas, palabra (secuencia), si es toroidal, si es palindromo.

Uso: python pocos_alfas.py --D 2 --ns 7 11 13 17 19 23 29 31
Salida: data/pocos_alfas_D{D}.json   {N: [[solucion, [alfas en orden de aparicion]], ...]}
"""
import argparse, json, os, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))


def sg(x, N):
    x %= N
    return x if x <= N // 2 else x - N


def build(N, D):
    out = []
    cols = [False] * N
    d1 = [False] * (2 * N)
    d2 = [False] * (2 * N)
    sol = [0] * N

    def rec(i, c, used):
        if i == N - 1:
            out.append((tuple(sol), tuple(used)))
            return
        cand = list(used)
        new = len(used) < D
        for s in (range(1, N) if new else cand):
            if not new and s not in used:
                continue
            c2 = (c + s) % N
            r = i + 1
            if cols[c2] or d1[r + c2] or d2[r - c2 + N]:
                continue
            cols[c2] = d1[r + c2] = d2[r - c2 + N] = True
            sol[r] = c2
            if s in used:
                rec(r, c2, used)
            else:
                rec(r, c2, used + [s])
            cols[c2] = d1[r + c2] = d2[r - c2 + N] = False

    for c0 in range(N):
        cols[c0] = d1[c0] = d2[N - c0] = True
        sol[0] = c0
        rec(0, c0, [])
        cols[c0] = d1[c0] = d2[N - c0] = False
    return out


def tor(s):
    N = len(s)
    return len({(i + c) % N for i, c in enumerate(s)}) == N and len({(i - c) % N for i, c in enumerate(s)}) == N


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--D', type=int, default=2)
    ap.add_argument('--ns', type=int, nargs='+', default=[7, 11, 13, 17, 19, 23])
    ap.add_argument('--max-seg', type=float, default=600)
    a = ap.parse_args()
    path = os.path.join(HERE, 'data', f'pocos_alfas_D{a.D}.json')
    res = json.load(open(path)) if os.path.exists(path) else {}
    print(f"soluciones con a lo sumo {a.D} alfas distintos")
    print(f"{'N':>3} | total | 1 alfa (toroidales) | exactamente {a.D}: toroidales / no toroidales | palindromos entre las no toroidales | seg")
    for N in a.ns:
        t0 = time.time()
        S = build(N, a.D)
        res[str(N)] = [[list(s), list(u)] for s, u in S]
        one = [s for s, u in S if len(u) == 1]
        ex = [(s, u) for s, u in S if len(u) == a.D]
        nt = [s for s, u in ex if not tor(s)]
        pal = 0
        for s in nt:
            w = [(s[i + 1] - s[i]) % N for i in range(N - 1)]
            pal += w == w[::-1]
        dt = time.time() - t0
        print(f"{N:3d} | {len(S):6d} | {len(one):5d} | {len(ex)-len(nt):5d} / {len(nt):5d} | {pal:5d} | {dt:.1f}", flush=True)
        json.dump(res, open(path, 'w'))
        if dt > a.max_seg:
            break
