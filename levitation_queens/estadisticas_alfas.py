"""
Estadisticas de alfas y tramos sobre TODAS las soluciones (N pequenos).

  alfa   = sigma(i+1) - sigma(i)  mod N, con signo en (-N/2, N/2]
  tramo  = racha maxima de alfas iguales consecutivos
Para cada solucion: n de tramos, n de alfas distintos, tramo mas largo.

Salidas por N:
  1. reparto de soluciones por numero de tramos
  2. alfas distintos, tramo mas largo, fraccion de pasos que estan en tramos largos (>= 2)
  3. que alfas se usan mas (|alfa|), comparado con el reparto uniforme
  4. (si hay datos de ondas) ondas minimas medias segun el n de tramos

Uso: python estadisticas_alfas.py 8 9 10 11 12 13
"""
import json, os, sys
from collections import Counter, defaultdict
import numpy as np
import min_waves as mw

HERE = os.path.dirname(os.path.abspath(__file__))


def sg(x, N):
    x %= N
    return x if x <= N // 2 else x - N


def describe(sol, N):
    a = [sg(sol[i + 1] - sol[i], N) for i in range(N - 1)]
    runs = []
    for x in a:
        if runs and runs[-1][0] == x:
            runs[-1][1] += 1
        else:
            runs.append([x, 1])
    return a, runs


def main(Ns):
    for N in Ns:
        sols = mw.solutions(N)
        K = {}
        f = os.path.join(HERE, 'data', f'waves_N{N}_nodeg.json')
        if os.path.exists(f):
            K = {tuple(r['solucion']): r['ondas_min'] for r in json.load(open(f))['soluciones']}
        R, D, L, inlong = [], [], [], []
        alf = Counter()
        byR = defaultdict(list)
        lens = Counter()
        for s in sols:
            a, runs = describe(s, N)
            R.append(len(runs)); D.append(len(set(a))); L.append(max(l for _, l in runs))
            inlong.append(sum(l for _, l in runs if l >= 2) / (N - 1))
            alf.update(abs(x) for x in a)
            lens.update(l for _, l in runs)
            if K:
                byR[len(runs)].append(K[s])
        print(f"===== N={N}: {len(sols)} soluciones, {N-1} pasos cada una")
        h = Counter(R)
        print("  1. tramos -> soluciones: " + "  ".join(f"{k}: {v}" for k, v in sorted(h.items())))
        print(f"     media {np.mean(R):.2f} tramos (maximo posible {N-1}); alfas distintos media {np.mean(D):.2f}")
        tl = sum(lens.values())
        print(f"  2. tramo mas largo: media {np.mean(L):.2f}, max {max(L)} | pasos dentro de tramos largos: {100*np.mean(inlong):.0f}%"
              f" | longitud de los tramos: " + "  ".join(f"{k}: {100*v/tl:.0f}%" for k, v in sorted(lens.items())))
        tot = sum(alf.values())
        vals = sorted(alf)
        unif = {v: (2 if (v != N / 2) else 1) / (N - 1) for v in range(1, N // 2 + 1)}
        print("  3. |alfa| -> % de los pasos (entre parentesis, veces respecto al azar): "
              + "  ".join(f"{v}: {100*alf[v]/tot:.0f}% ({alf[v]/tot/unif[v]:.1f}x)" for v in vals))
        if K:
            print("  4. tramos -> ondas minimas medias: " + "  ".join(f"{k}: {np.mean(v):.2f}" for k, v in sorted(byR.items()))
                  + f"   (correlacion {np.corrcoef(R, [K[s] for s in sols])[0, 1]:.2f})")


if __name__ == '__main__':
    main([int(a) for a in sys.argv[1:]] or [8, 9, 10, 11, 12, 13])
