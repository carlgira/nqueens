"""Toro con un liquido que fluye hacia el frio: una particula que resuelve N-reinas.

Coordenadas planas del toro: r = fila continua en [0, N) (angulo largo,
theta = 2 pi r / N) y x = columna continua (angulo corto, phi = 2 pi x / N).

  dr/dt = 1                            el rio arrastra: una fila por unidad de tiempo
  dx/dt = u(r, x, t)                   el liquido corre hacia lo frio de la fila que viene

Mapa de calor H(r, c): cada una de las ultimas N-1 reinas depositadas (la
"ventana": una vuelta menos la fila que se esta cruzando) calienta su columna y
sus dos diagonales -> rayas verticales y a 45 grados sobre el toro. Encima, una
memoria lenta M(fila, c) que se magnetiza en cada celda visitada y decae con
tasa lam por fila.

  H_c = A * (ataques suaves sobre la celda (fila_siguiente, c)) + beta * M(fila_siguiente, c)

Campo de velocidad (suave en el toro): pesos de Boltzmann sobre las columnas
de la fila que viene, w_c = exp(-(H_c + eps * d(c, x+alpha)^2) / T), y la
corriente gira la fase de la particula hacia la fase media ponderada:

  Z = sum_c w_c e^{i phi_c},   u = kappa * (|Z|/sum w) * (N/2pi) * sin(arg Z - phi)

alpha es la pendiente del rio (columnas por fila): sin calor la particula
traza la geodesica de pendiente alpha (la construccion mod-N). El calor la
dobla; la memoria la empuja a no repetir.

Al cruzar el centro de cada fila se deposita una reina en round(x). Cada vuelta
completa (r de 0 a N) es una permutacion candidata.
"""
import numpy as np


def is_solution(cols, toroidal=False):
    n = len(cols)
    if len(set(cols)) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            d = cols[j] - cols[i]
            if toroidal:
                if (d - (j - i)) % n == 0 or (d + (j - i)) % n == 0:
                    return False
            elif abs(d) == j - i:
                return False
    return True


class TorusFlow:
    def __init__(self, n=8, alpha=2.0, A=1.0, s=0.3, beta=0.3, lam=0.05,
                 T=0.15, eps=0.02, kappa=None, toroidal=False, x0=0.0,
                 dt=0.02):
        self.n, self.alpha, self.A, self.s = n, alpha, A, s
        self.beta, self.lam, self.T, self.eps = beta, lam, T, eps
        self.kappa = 3.0 * n if kappa is None else kappa
        self.toroidal, self.dt = toroidal, dt
        self.t, self.x = 0.0, float(x0)
        self.win = np.full(n, np.nan)       # columna continua depositada por fila
        self.mem = np.zeros((n, n))          # magnetizacion por celda
        self.cols = np.arange(n)
        self.phases = np.exp(2j * np.pi * self.cols / n)
        self.lap_cols, self.laps = [], []
        self._ver, self._cache = 0, (None, None)

    def wrap(self, u):
        return (u + self.n / 2) % self.n - self.n / 2

    def label(self, x):
        return (x + 0.5) % self.n - 0.5

    # --- mapa de calor ---------------------------------------------------
    def heat_row(self, row, xs=None):
        """Calor sobre la fila `row` evaluado en columnas xs (por defecto enteras)."""
        n, s = self.n, self.s
        xs = self.cols.astype(float) if xs is None else np.asarray(xs, float)
        h = np.zeros_like(xs)
        for j in range(n):
            xj = self.win[j]
            if np.isnan(xj) or j == row:
                continue
            dr = row - j
            if self.toroidal:
                us = [self.wrap(xs - xj - k * dr) for k in (0, 1, -1)]
            else:  # columnas envueltas, diagonales sin envolver (falla en la costura)
                xl, xjl = self.label(xs), self.label(xj)
                us = [self.wrap(xs - xj), xl - xjl - dr, xl - xjl + dr]
            for u in us:
                h += np.exp(-u * u / (2 * s * s))
        return self.A * h + self.beta * self.mem[row, np.rint(xs).astype(int) % n]

    def next_row(self, r=None):
        r = self.t if r is None else r
        rr = r % self.n
        return int(np.floor(rr - 0.5) + 1) % self.n   # siguiente centro de fila

    def velocity(self, x, row):
        n = self.n
        if self._cache[0] != (row, self._ver):   # el calor solo cambia al depositar
            self._cache = ((row, self._ver), self.heat_row(row))
        h = self._cache[1]
        d = self.wrap(self.cols - (x + self.alpha * self.time_to_center()))
        e = (h + self.eps * d * d) / self.T
        w = np.exp(-(e - e.min()))
        Z = (w * self.phases).sum() / w.sum()
        phi = 2 * np.pi * x / n
        return self.kappa * abs(Z) * n / (2 * np.pi) * np.sin(np.angle(Z) - phi)

    def time_to_center(self):
        rr = self.t % self.n
        return (np.floor(rr - 0.5) + 1.5 - rr)

    # --- integracion -----------------------------------------------------
    def step(self):
        n, dt = self.n, self.dt
        row = self.next_row()
        # RK2 (punto medio) con la fila objetivo fija dentro del paso
        k1 = self.velocity(self.x, row)
        k2 = self.velocity(self.x + 0.5 * dt * k1, row)
        c_old = np.floor(self.t - 0.5)
        self.x = (self.x + dt * k2) % n
        self.t += dt
        if np.floor(self.t - 0.5) != c_old:                    # cruzamos el centro de `row`
            c = int(np.rint(self.x)) % n
            self.win[row] = self.x
            self.mem *= np.exp(-self.lam)
            self.mem[row, c] += 1.0
            self._ver += 1
            self.lap_cols.append(c)
            if row == n - 1:
                if len(self.lap_cols) == n:
                    self.laps.append(tuple(self.lap_cols))
                self.lap_cols = []

    def run(self, laps):
        target = len(self.laps) + laps
        while len(self.laps) < target:
            self.step()
        return self.laps
