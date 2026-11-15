"""
Construir DIRECTAMENTE las soluciones hechas de pocos tramos de paso constante.

Tramo = filas consecutivas con el mismo paso  s = sigma(i+1) - sigma(i)  (mod N).
Se buscan todas las soluciones con a lo sumo R tramos (R-1 cambios de paso) por
backtracking: en cada fila o se sigue con el paso actual o se cambia (si quedan cambios).
No hace falta enumerar las Q(N) soluciones.

Firma de una solucion: (pasos de cada tramo, longitudes de cada tramo).

Uso: python tramos.py --R 3 --desde 5 --hasta 30
Salida: data/tramos_R{R}.json   {N: [[solucion, pasos, longitudes], ...]}
"""
import argparse, json, os, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.setrecursionlimit(10000)


def build(N, R):
    out = []
    cols = [False] * N
    d1 = [False] * (2 * N)
    d2 = [False] * (2 * N)
    sol = [0] * N

    def place(i, c):
        if cols[c] or d1[i + c] or d2[i - c + N]:
            return False
        cols[c] = d1[i + c] = d2[i - c + N] = True
        sol[i] = c
        return True

    def unplace(i, c):
        cols[c] = d1[i + c] = d2[i - c + N] = False

    def rec(i, c, s, changes, steps, lens):
        if i == N - 1:
            out.append((tuple(sol), tuple(steps), tuple(lens)))
            return
        # seguir con el mismo paso
        if s is not None:
            c2 = (c + s) % N
            if place(i + 1, c2):
                lens[-1] += 1
                rec(i + 1, c2, s, changes, steps, lens)
                lens[-1] -= 1
                unplace(i + 1, c2)
        # cambiar de paso (o elegir el primero)
        if s is None or changes < R - 1:
            for s2 in range(1, N):              # 0 repite columna; el resto lo decide place() (un +-1 que da la vuelta es valido)
                if s2 == s:
                    continue
                c2 = (c + s2) % N
                if place(i + 1, c2):
                    steps.append(s2); lens.append(1)
                    rec(i + 1, c2, s2, changes + (s is not None), steps, lens)
                    steps.pop(); lens.pop()
                    unplace(i + 1, c2)

    for c0 in range(N):
        place(0, c0)
        rec(0, c0, None, 0, [], [])
        unplace(0, c0)
    return out


def toroidal(sol, N):
    return len({(i + c) % N for i, c in enumerate(sol)}) == N and len({(i - c) % N for i, c in enumerate(sol)}) == N


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--R', type=int, default=3)
    ap.add_argument('--desde', type=int, default=5)
    ap.add_argument('--hasta', type=int, default=24)
    ap.add_argument('--max-seg', type=float, default=600)
    a = ap.parse_args()
    path = os.path.join(HERE, 'data', f'tramos_R{a.R}.json')
    res = {int(k): v for k, v in json.load(open(path)).items()} if os.path.exists(path) else {}   # acumula
    print(f"soluciones con a lo sumo {a.R} tramos de paso constante (mod N)")
    print(f"{'N':>3} {'N%6':>4} {'total':>7} | por n de tramos exacto | de 1 tramo toroidales | seg")
    for N in range(a.desde, a.hasta + 1):
        t0 = time.time()
        sols = build(N, a.R)
        byr = Counter(len(st) for _, st, _ in sols)
        res[N] = [[list(s), list(st), list(ln)] for s, st, ln in sols]
        dt = time.time() - t0
        print(f"{N:3d} {N%6:4d} {len(sols):7d} | {dict(sorted(byr.items()))} | "
              f"{sum(toroidal(s, N) for s, st, _ in sols if len(st) == 1)} | {dt:.1f}", flush=True)
        json.dump(res, open(path, 'w'))
        if dt > a.max_seg:
            break
