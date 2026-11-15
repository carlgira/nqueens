"""
Contar por NIVELES de filas y predecir el total.

P_k(N) = n de formas de colocar k reinas sin ataques en las k primeras filas.
    P_0 = 1, P_1 = N, ..., P_N = Q(N)  (soluciones)
Factor de cada nivel:  b_k = P_{k+1} / P_k  = casillas libres de media en la fila siguiente.

Hipotesis (curva universal):  b_k(N) ~ N * g(x) + h(x),  x = k/N
    g(x): la forma limite;  h(x): correccion de tablero finito.
Se ajustan g y h con los N pequenos (regresion lineal en N para cada x, interpolando) y se
predice  ln Q(N) = suma_k ln b_k  para N mayores, comparando con los valores conocidos.

Uso: python perfil.py --hasta 14
"""
import argparse, json, math, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Q = {4: 2, 5: 10, 6: 4, 7: 40, 8: 92, 9: 352, 10: 724, 11: 2680, 12: 14200, 13: 73712, 14: 365596,
     15: 2279184, 16: 14772512, 17: 95815104, 18: 666090624, 19: 4968057848, 20: 39029188884,
     21: 314666222712, 22: 2691008701644, 23: 24233937684440, 24: 227514171973736,
     25: 2207893435808352, 26: 22317699616364044, 27: 234907967154122528}


def profile(N):
    P = [0] * (N + 1)
    full = (1 << N) - 1

    def rec(k, cols, d1, d2):
        P[k] += 1
        if k == N:
            return
        free = full & ~(cols | d1 | d2)
        while free:
            bit = free & -free
            free ^= bit
            rec(k + 1, cols | bit, ((d1 | bit) << 1) & full, (d2 | bit) >> 1)
    rec(0, 0, 0, 0)
    return P


def predict(profiles, Ntrain, Ntest):
    """b_k(N) = N g(x) + h(x): para cada x se ajusta una recta en N con los perfiles de Ntrain."""
    xs = np.linspace(0, 1, 201)[:-1]
    B = []
    for N in Ntrain:
        P = profiles[N]
        b = np.array([P[k + 1] / P[k] for k in range(N)])
        B.append(np.interp(xs, np.arange(N) / N, b))        # b como funcion de x = k/N
    B = np.array(B)
    Nv = np.array(Ntrain, float)
    A = np.c_[Nv, np.ones(len(Nv))]
    coef, *_ = np.linalg.lstsq(A, B, rcond=None)             # filas: g(x), h(x)
    g, h = coef
    out = {}
    for N in Ntest:
        x = np.arange(N) / N
        b = N * np.interp(x, xs, g) + np.interp(x, xs, h)
        b = np.maximum(b, 1e-9)
        out[N] = float(np.sum(np.log(b)))
    return out, xs, g, h


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hasta', type=int, default=14)
    a = ap.parse_args()
    path = os.path.join(HERE, 'data', 'perfil.json')
    prof = {int(k): v for k, v in json.load(open(path)).items()} if os.path.exists(path) else {}
    for N in range(4, a.hasta + 1):
        if N not in prof:
            t0 = time.time()
            prof[N] = profile(N)
            assert prof[N][N] == Q[N]
            json.dump(prof, open(path, 'w'))
            print(f"  perfil N={N} calculado ({time.time()-t0:.0f}s)", flush=True)
    print("\nNiveles P_k (colocaciones validas de k filas):")
    for N in (8, 10, 12, 14):
        if N in prof:
            P = prof[N]
            print(f"  N={N}: " + " ".join(str(p) for p in P) + f"   | maximo en k={int(np.argmax(P))} | total de nodos {sum(P)}")
    print("\nFactor por nivel dividido por N,  b_k/N,  en x = k/N  (si la curva es universal, las filas se parecen):")
    xs0 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("   N  | " + " ".join(f"x={x:.1f}" for x in xs0))
    for N in sorted(prof):
        if N < 8:
            continue
        P = prof[N]
        b = np.array([P[k + 1] / P[k] for k in range(N)]) / N
        print(f"  {N:3d} | " + " ".join(f"{np.interp(x, np.arange(N)/N, b):5.3f}" for x in xs0))
    train = [N for N in sorted(prof) if N >= 8]
    test = list(range(max(train) + 1, 28))
    pred, xs, g, h = predict(prof, train, test)
    print(f"\nPrediccion de Q(N) con la curva ajustada en N = {train[0]}..{train[-1]}:")
    print(f"{'N':>3} | {'predicho':>12} | {'real':>12} | predicho/real | error en ln Q por reina")
    for N in test:
        p = math.exp(pred[N])
        print(f"{N:3d} | {p:12.4g} | {Q[N]:12.4g} | {p/Q[N]:8.3f} | {(pred[N]-math.log(Q[N]))/N:+.4f}")
    # constante de crecimiento: ln Q / N - ln N
    print("\nln Q(N)/N - ln N  (tiende a -alfa; Simkin 2021: alfa ~ 1,94):")
    print("  " + "  ".join(f"{N}: {math.log(Q[N])/N - math.log(N):.3f}" for N in (8, 12, 16, 20, 24, 27)))
    json.dump({'x': xs.tolist(), 'g': g.tolist(), 'h': h.tolist()}, open(os.path.join(HERE, 'data', 'perfil_curva.json'), 'w'))


# ---------------------------------------------------------------- prediccion robusta (escala log)
def predict_log(prof, train, Ns):
    """ln(b_k/N) = G(x) + C(x)/N,  x = (k+1/2)/N.  Ajuste por minimos cuadrados en 1/N para cada x."""
    xs = np.linspace(0, 1, 401)[:-1]
    Y = np.array([np.interp(xs, (np.arange(N) + 0.5) / N,
                            np.log(np.array([prof[N][k + 1] / prof[N][k] for k in range(N)]) / N)) for N in train])
    A = np.c_[np.ones(len(train)), 1 / np.array(train, float)]
    (G, C), *_ = np.linalg.lstsq(A, Y, rcond=None)
    out = {}
    for N in Ns:
        x = (np.arange(N) + 0.5) / N
        out[N] = N * math.log(N) + float(np.sum(np.interp(x, xs, G) + np.interp(x, xs, C) / N))
    return out
