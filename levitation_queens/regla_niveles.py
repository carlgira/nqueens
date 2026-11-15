"""
Regla aproximada de crecimiento de los niveles de fusion.

Si ya tenemos H de las Q soluciones (como si fueran una muestra al azar), una solucion T sale
en el siguiente nivel si para alguna plantilla sus dos mitades estan presentes:

    P(mitad presente) = 1 - (1 - H/Q)^c        c = n de OTRAS soluciones que tienen esa mitad
                                                (T no cuenta: todavia no la tenemos)
    P(T por la plantilla g) = P(abierta) * P(cerrada)
    P(T en el nivel siguiente) = 1 - prod_g (1 - P_g)

    H' = H + (Q - H) * media_T [ P(T en el nivel siguiente) ]

Dos versiones:
  'detallada' : usa los c reales de cada solucion y plantilla (necesita las soluciones)
  'numeros'   : usa solo c medio por tamano de plantilla (una tabla pequena de numeros)

Se compara con los niveles reales (niveles.py, sin cerrar por simetrias aqui: ver nota).

Uso: python regla_niveles.py 11 13
"""
import sys, json, os
from collections import defaultdict
import numpy as np
import min_waves as mw
from fusion import wave_gates

HERE = os.path.dirname(os.path.abspath(__file__))


def share_counts(N):
    """para cada plantilla: (tamano k, c_abierta por solucion, c_cerrada por solucion)"""
    sols = mw.solutions(N)
    S = np.array(sols, dtype=np.int64)
    rng = np.random.default_rng(0)
    W = rng.integers(1, 2 ** 40, size=N, dtype=np.int64)
    X = (S + 1) * W
    out = []
    for G in wave_gates(N):
        m = np.zeros(N, bool); m[list(G)] = True
        co = np.unique(X[:, m].sum(1), return_inverse=True, return_counts=True)
        cc = np.unique(X[:, ~m].sum(1), return_inverse=True, return_counts=True)
        out.append((int(m.sum()), co[2][co[1]] - 1, cc[2][cc[1]] - 1))
    return sols, out


def step_detailed(H, Q, SC):
    f = H / Q
    logq = np.zeros(Q)
    for k, co, cc in SC:
        p = (1 - (1 - f) ** co) * (1 - (1 - f) ** cc)
        logq += np.log1p(-np.minimum(p, 1 - 1e-12))
    return H + (Q - H) * float(np.mean(1 - np.exp(logq)))


def step_numbers(H, Q, table):
    """table: lista de (n de plantillas, c medio abierta, c medio cerrada)"""
    f = H / Q
    logq = 0.0
    for ng, mo, mc in table:
        p = (1 - (1 - f) ** mo) * (1 - (1 - f) ** mc)
        logq += ng * np.log1p(-min(p, 1 - 1e-12))
    return H + (Q - H) * (1 - np.exp(logq))


def run(step, H0, Q, arg, maxlev=60):
    seq = [H0]
    for _ in range(maxlev):
        H = step(seq[-1], Q, arg)
        if H - seq[-1] < 0.5:
            break
        seq.append(H)
    return seq


if __name__ == '__main__':
    for N in [int(x) for x in sys.argv[1:]] or [11, 13]:
        sols, SC = share_counts(N)
        Q = len(sols)
        by = defaultdict(list)
        for k, co, cc in SC:
            by[k].append((co.mean(), cc.mean()))
        table = [(len(v), float(np.mean([a for a, b in v])), float(np.mean([b for a, b in v]))) for k, v in sorted(by.items())]
        print(f"===== N={N}: Q = {Q}, {len(SC)} plantillas")
        print("   tabla de numeros (plantillas por tamano, c medio de la mitad pequena, c medio de la grande):")
        for (k, v), t in zip(sorted(by.items()), table):
            print(f"      {k} filas: {t[0]} plantillas, c = {t[1]:.2f} y {t[2]:.3f}")
        for name in ('tramo', 'reglas'):
            f = os.path.join(HERE, 'data', f'niveles_N{N}_{name}.npy')
            if not os.path.exists(f):
                continue
            lev = np.load(f)
            real = np.cumsum([int((lev == L).sum()) for L in range(lev.max() + 1)])
            H0 = int(real[0])
            d = run(step_detailed, H0, Q, SC)
            n = run(step_numbers, H0, Q, table)
            fmt = lambda s: " ".join(f"{int(round(x))}" for x in s[:12]) + (" ..." if len(s) > 12 else "")
            print(f"   semilla '{name}' ({H0}):")
            print(f"      real (acumulado)        : {fmt(real)}   -> {int(real[-1])} en {len(real)-1} niveles")
            print(f"      regla detallada         : {fmt(d)}   -> {int(round(d[-1]))} en {len(d)-1} niveles")
            print(f"      regla con solo numeros  : {fmt(n)}   -> {int(round(n[-1]))} en {len(n)-1} niveles")
