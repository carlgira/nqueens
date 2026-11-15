"""
Spiral N-Queens
===============

Núcleo mínimo extraído de `super_queens.py`.

La idea: para un N primo se construye una rejilla POLAR de N x N puntos
(N ángulos equiespaciados x N radios que crecen según una espiral logarítmica
basada en el número áureo `phi`). Si se traza una espiral logarítmica con una
constante `c` que es una *fracción* de `phi`, la espiral pasa exactamente por
un punto de cada radio, y esos N puntos resultan ser una solución válida del
problema de las N reinas.

Recorriendo las distintas fracciones de `phi` (parámetro `d`), los N desfases
angulares y las 2 direcciones de la espiral se obtienen todas las soluciones
"espirales" del tablero. Para N=7 eso da 2*7*2 = 28 soluciones.
"""

import math
import numpy as np

# Número áureo
phi = (1 + math.sqrt(5)) / 2

# Dos espirales logarítmicas (una en cada sentido de giro) en forma polar:
#   spiral(c, theta, l) -> radio     ;     spiral_inv(c, r, l) -> theta
spiral1 = lambda c, t, l: math.pow(c, t - 12 * math.pi + l)
spiral1_inv = lambda c, r, l: math.log(r, c) + 12 * math.pi - l
spiral2 = lambda c, t, l: math.pow(c, -(t - 2 * math.pi + l))
spiral2_inv = lambda c, r, l: 2 * math.pi - math.log(r, c) - l


def is_prime(n):
    return n > 1 and all(n % j for j in range(2, int(n ** 0.5) + 1))


def polar_2_cartesian(theta, r, x0=0.0, y0=0.0):
    return r * np.cos(theta) + x0, r * np.sin(theta) + y0


class SpiralQueens:
    def __init__(self, n):
        self.n = n

        # Constante base de la espiral que define los radios de la rejilla.
        c1 = math.pow(phi, 2 / math.pi)

        # Ángulos (rayos) y radios (anillos) de la rejilla polar.
        self.a = np.array([(2 * math.pi * i) / n for i in range(n)])
        self.r = np.array(
            [spiral1(c1, 10 * math.pi + 2 * math.pi / n, aa) for aa in self.a]
        )

        # Rejilla polar de N x N puntos, en coordenadas cartesianas.
        # mp[i][j] = punto en el ángulo a[i] y el radio r[j].
        self.mp = np.zeros((n, n, 2))
        for i in range(n):
            for j in range(n):
                self.mp[i][j] = [
                    round(self.r[j] * math.cos(self.a[i]), 8),
                    round(self.r[j] * math.sin(self.a[i]), 8),
                ]
        self.mp_flatten = self.mp.reshape(-1, 2).tolist()

    def get_sol(self, c, a=0.0):
        """Devuelve las 2 soluciones (espiral hacia un lado / hacia el otro) que
        pasan por la rejilla polar con constante `c` y desfase angular `a`.

        Cada solución es un array de pares [indice_radio, indice_angulo]. La
        columna [:, 1] (los índices de ángulo, ordenados por radio) es la
        permutación que representa la solución de N reinas.
        """
        app = [spiral1_inv(c, r, a) for r in self.r]
        ass = [spiral2_inv(c, r, a) for r in self.r]

        xyp = [[round(r * math.cos(t), 8), round(r * math.sin(t), 8)]
               for r, t in zip(self.r, app)]
        xys = [[round(r * math.cos(t), 8), round(r * math.sin(t), 8)]
               for r, t in zip(self.r, ass)]

        s1, s2 = [], []
        for xypi, xysi in zip(xyp, xys):
            try:
                k = self.mp_flatten.index(xypi)
                s1.append([k % self.n, k // self.n])
            except ValueError:
                pass
            try:
                k = self.mp_flatten.index(xysi)
                s2.append([k % self.n, k // self.n])
            except ValueError:
                pass

        return np.array(s1), np.array(s2)

    def validate(self, sol):
        """Comprueba que `sol` es una solución válida de N reinas."""
        sol = list(sol)
        n = len(sol)
        if n != self.n:
            return False
        if len(set(sol)) != n:                       # una reina por fila
            return False
        if len(set(sol[i] + i for i in range(n))) != n:   # diagonales '/'
            return False
        if len(set(sol[i] - i for i in range(n))) != n:   # diagonales '\'
            return False
        return True

    def init_sols_label(self):
        """Todas las soluciones espirales para este N, cada una etiquetada con
        (solucion, d, indice_angulo, direccion).

        direccion = +1 : espiral 1 (un sentido de giro)
        direccion = -1 : espiral 2 (sentido opuesto)

        Para N=7 devuelve 2*7*2 = 28 entradas.
        """
        max_c = int(self.n / 2)
        sols = []
        for d in range(1, max_c):
            c = math.pow(phi, 1 / (d * math.pi))
            for i, a in enumerate(self.a):
                s1, s2 = self.get_sol(c, a)
                sols.append((s1[:, 1], d, i, +1))
                sols.append((s2[:, 1], d, i, -1))
        return sols


if __name__ == "__main__":
    sq = SpiralQueens(7)
    labels = sq.init_sols_label()
    print(f"N=7 -> {len(labels)} soluciones espirales")
    valid = sum(sq.validate(sol) for sol, *_ in labels)
    print(f"Válidas: {valid}/{len(labels)}")
    for sol, d, i, direction in labels:
        tag = "s1" if direction == 1 else "s2"
        print(f"  d={d} ang={i} {tag}: {list(sol)}  {'OK' if sq.validate(sol) else 'X'}")
