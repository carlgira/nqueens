"""
¿Por que nunca se pasa de una toroidal a otra moviendo 3 o 5 reinas?

Si se recolocan k reinas (filas r_i, columnas c_i) y sale otra solucion, las reinas movidas ocupan las
mismas columnas y las mismas diagonales de los dos tipos. Entonces hay tres permutaciones de las k reinas:
   sigma: la fila i recibe la columna de la reina sigma(i)
   tau:   y cae en la diagonal "/" que tenia la reina tau(i):   c_sigma(i) + r_i = c_tau(i) + r_tau(i)
   rho:   y en la diagonal "\\" que tenia la reina rho(i):       c_sigma(i) - r_i = c_rho(i) - r_rho(i)
En cada i los cuatro valores i, sigma(i), tau(i), rho(i) son distintos (si dos coinciden, dos reinas
comparten fila o columna). O sea (id, sigma, tau, rho) es un rectangulo latino de 4 x k: hace falta k >= 4.
Eso prueba k = 3. Para k = 5 se recorren todos los rectangulos; cada uno da 2k ecuaciones lineales en
(r, c). Si de ellas se deduce r_i = r_j o c_i = c_j (dos reinas en la misma fila o columna), ese
rectangulo es imposible. El programa busca, para cada rectangulo, un certificado entero
   suma n_e * ecuacion_e = D * (r_i - r_j)   (o c_i - c_j),  D != 0,
que vale para todo primo que no divida a D.

Uso: ../venv/bin/python intercambio_imposible.py [k ...]     (por defecto 3 4 5 6)
"""
import sys
from fractions import Fraction
from itertools import permutations, combinations
from math import lcm, gcd
from collections import Counter


def rref(M):
    """forma escalonada reducida sobre Q de las filas de M; devuelve (filas, pivotes)."""
    M = [[Fraction(x) for x in row] for row in M]; piv = []; f = 0
    for col in range(len(M[0])):
        k = next((i for i in range(f, len(M)) if M[i][col] != 0), None)
        if k is None: continue
        M[f], M[k] = M[k], M[f]
        pv = M[f][col]; M[f] = [x / pv for x in M[f]]
        for i in range(len(M)):
            if i != f and M[i][col] != 0:
                q = M[i][col]; M[i] = [x - q * y for x, y in zip(M[i], M[f])]
        piv.append(col); f += 1
    return M[:f], piv


def certificado(ecs, k):
    """menor D tal que D*(x_i - x_j) es combinacion entera de las ecuaciones, o None."""
    # se trabaja con la matriz ampliada [ecuaciones | identidad] para recuperar la combinacion
    n = len(ecs)
    A = [list(e) + [1 if i == j else 0 for j in range(n)] for i, e in enumerate(ecs)]
    R, piv = rref(A)
    R = [row for row, pc in zip(R, piv) if pc < 2 * k]; piv = [pc for pc in piv if pc < 2 * k]
    mejor = None
    for base in (0, k):
        for i, j in combinations(range(k), 2):
            f = [Fraction(0)] * (2 * k); f[base + i] = Fraction(1); f[base + j] = Fraction(-1)
            comb = [Fraction(0)] * n; g = f[:]
            for row, pc in zip(R, piv):
                if g[pc] != 0:
                    q = g[pc]; g = [x - q * y for x, y in zip(g, row[:2 * k])]
                    comb = [x + q * y for x, y in zip(comb, row[2 * k:])]
            if any(g): continue                              # f no se deduce de las ecuaciones
            D = lcm(*[c.denominator for c in comb])
            D //= gcd(D, *[int(c * D) for c in comb if c != 0]) if any(comb) else 1
            if mejor is None or D < mejor: mejor = D
    return mejor


def rectangulos(k):
    ids = tuple(range(k))
    der = [s for s in permutations(range(k)) if all(s[i] != i for i in range(k))]
    for s in der:
        for t in der:
            if any(t[i] == s[i] for i in range(k)): continue
            for r in der:
                if any(r[i] == s[i] or r[i] == t[i] for i in range(k)): continue
                yield s, t, r


if __name__ == "__main__":
    for k in (list(map(int, sys.argv[1:])) or [3, 4, 5, 6]):
        tot = 0; libres = 0; Ds = Counter()
        for s, t, r in rectangulos(k):
            tot += 1
            ecs = []
            for i in range(k):
                e = [0] * (2 * k); e[k + s[i]] += 1; e[i] += 1; e[k + t[i]] -= 1; e[t[i]] -= 1; ecs.append(e)
                e = [0] * (2 * k); e[k + s[i]] += 1; e[i] -= 1; e[k + r[i]] -= 1; e[r[i]] += 1; ecs.append(e)
            D = certificado(ecs, k)
            if D is None: libres += 1
            else: Ds[D] += 1
        if tot == 0:
            print(f"k={k}: no existe ningun rectangulo latino 4 x {k}  ->  imposible para todo p"); continue
        primos = sorted({q for D in Ds for q in range(2, D + 1) if D % q == 0 and all(q % z for z in range(2, q))})
        print(f"k={k}: {tot} rectangulos; {tot - libres} imposibles con certificado (valores de D: {dict(sorted(Ds.items()))}),"
              f" {libres} sin obstaculo lineal")
        if libres == 0:
            print(f"      -> mover exactamente {k} reinas es imposible para todo primo p que no divida a ningun D"
                  f" (primos que dividen a algun D: {primos}), es decir para todo p >= {max(primos + [2]) + 1}")
