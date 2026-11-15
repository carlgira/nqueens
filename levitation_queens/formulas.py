"""
¿Los niveles por filas P_k(N) tienen formula combinatoria?

P_k(N) = colocaciones sin ataques de k reinas en las k primeras filas de un tablero N x N.
Para k fijo se busca un polinomio en N de grado k con coeficientes enteros (interpolacion
exacta con fracciones), ajustado con unos N y COMPROBADO en los demas.

Datos: data/perfil_c.txt  (./perfil_c N)

Uso: python formulas.py
"""
import os
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    P = {}
    for line in open(os.path.join(HERE, 'data', 'perfil_c.txt')):
        v = [int(x) for x in line.split()]
        if v:
            P[len(v) - 1] = v
    return P


def interp(xs, ys):
    """coeficientes (grado creciente) del polinomio que pasa por los puntos, exactos"""
    n = len(xs)
    A = [[Fraction(x) ** j for j in range(n)] + [Fraction(y)] for x, y in zip(xs, ys)]
    for i in range(n):
        p = next(r for r in range(i, n) if A[r][i] != 0)
        A[i], A[p] = A[p], A[i]
        A[i] = [v / A[i][i] for v in A[i]]
        for r in range(n):
            if r != i and A[r][i] != 0:
                A[r] = [a - A[r][i] * b for a, b in zip(A[r], A[i])]
    return [A[i][n] for i in range(n)]


def poly_str(c):
    out = []
    for j in range(len(c) - 1, -1, -1):
        if c[j] == 0:
            continue
        v = c[j]
        s = f"{'+' if v > 0 else '-'} {abs(v)}"
        out.append(s + ('' if j == 0 else ' N' if j == 1 else f' N^{j}'))
    return ' '.join(out).lstrip('+ ')


if __name__ == '__main__':
    P = load()
    Ns = sorted(P)
    for k in range(1, 8):
        found = False
        for start in range(k, Ns[-1] - k - 1):          # la formula puede valer solo desde cierto N
            xs = [N for N in Ns if N >= start][:k + 1]
            if len(xs) < k + 1:
                break
            rest = [N for N in Ns if N > xs[-1]]
            if len(rest) < 2:
                break
            c = interp(xs, [P[N][k] for N in xs])
            ok = all(sum(cj * N ** j for j, cj in enumerate(c)) == P[N][k] for N in rest)
            if ok:
                print(f"k={k}: P_k(N) = {poly_str(c)}      exacto para N >= {start} (comprobado en {len(rest)} valores mas, hasta N={Ns[-1]})")
                found = True
                break
        if not found:
            print(f"k={k}: no hay un polinomio de grado {k} que cuadre con los datos hasta N={Ns[-1]}")
