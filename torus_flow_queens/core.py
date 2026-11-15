"""Toro con un liquido que fluye hacia el frio, en tiempo continuo.

Coordenadas planas del toro: r = fila continua (angulo largo, theta = 2 pi r/N)
y x = columna continua (tubo, phi = 2 pi (x+1/2)/N). Todo cambia de forma suave:

  dr/dt = 1                                  el rio: una fila por unidad de tiempo
  dx/dt = v,  dv/dt = (u(r, x, t) - v)/tau   la particula se deja llevar por el liquido

Campo de calor H(r, x, t), continuo en t:
  - Cada vez que la particula cruza el centro de una fila deja una reina en su x.
    La reina nace con intensidad 0 (sube entre edad 0.1 y 0.6), sus tres rayas
    (columna y dos diagonales) crecen hacia fuera a velocidad c, y se apaga
    entre edad N-1 y N-0.55, justo antes de que la particula vuelva a su fila.
  - Memoria m(fila, c): se carga continuamente cerca del centro de cada fila
    (pulso gaussiano, reparto suave entre columnas), decae con tasa lam por
    unidad de tiempo, y se interpola linealmente entre filas.

Corriente del liquido: gira la fase hacia la columna fria de la fila que viene,
pasando suavemente de una fila a la siguiente (ver current):

  w_c = exp(-E_c / T),  E_c = mezcla suave de H(fila, c) + eps * d(c, x + alpha*(tiempo al centro))^2
  u   = kappa * |Z| * (N/2pi) * sin(arg Z - phi),   Z = sum_c w_c e^{i phi_c}

Una vuelta completa = N reinas (columnas round(x) en los centros de fila).
El modelo anterior, con saltos en cada deposito, esta en core_pulsos.py.
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


def smoothstep(u, a, b):
    t = np.clip((u - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class TorusFlow:
    def __init__(self, n=8, alpha=2.0, A=1.0, s=0.3, beta=0.3, lam=0.05,
                 T=0.15, eps=0.02, kappa=None, tau=0.12, c=3.0,
                 toroidal=False, x0=0.0, dt=0.02):
        self.n, self.alpha, self.A, self.s = n, alpha, A, s
        self.beta, self.lam, self.T, self.eps = beta, lam, T, eps
        self.kappa = 3.0 * n if kappa is None else kappa
        self.tau, self.c = tau, c
        self.toroidal, self.dt = toroidal, dt
        self.t, self.x, self.v = 0.0, float(x0), 0.0
        self.q_t = np.full(n, -1e9)       # instante de deposito por fila
        self.q_x = np.zeros(n)             # columna continua depositada
        self.mem = np.zeros((n, n))
        self.cols = np.arange(n, dtype=float)
        self.phases = np.exp(2j * np.pi * self.cols / n)
        self.lap_cols, self.laps = [], []

    def wrap(self, u):
        return (u + self.n / 2) % self.n - self.n / 2

    def label(self, x):
        return (x + 0.5) % self.n - 0.5

    # --- calor -------------------------------------------------------------
    def strength(self, age):
        n = self.n
        return smoothstep(age, 0.1, 0.6) * (1 - smoothstep(age, n - 1.0, n - 0.55))

    def heat(self, r, xs, t=None, skip=None):
        """Calor en la fila continua r (mod N) sobre columnas xs, en el instante t.
        skip: fila cuya propia reina no cuenta (la que se esta decidiendo)."""
        n, s2 = self.n, 2 * self.s ** 2
        t = self.t if t is None else t
        xs = np.asarray(xs, float)
        rm = r % n
        age = t - self.q_t
        S = self.strength(age)
        h = np.zeros_like(xs)
        # en el tablero clasico las diagonales terminan en el borde: se apagan cerca de la costura
        edge = 1.0 if self.toroidal else smoothstep(rm, 0, 0.35) * (1 - smoothstep(rm, n - 0.35, n))
        for j in np.nonzero(S > 1e-4)[0]:
            if j == skip:
                continue
            d = rm - (j + 0.5)
            if self.toroidal:
                d = self.wrap(d)
            reach = smoothstep(self.c * age[j] - abs(d), 0.0, 0.5)   # las rayas crecen a velocidad c
            u0 = self.wrap(xs - self.q_x[j])
            if self.toroidal:
                u1, u2 = self.wrap(u0 - d), self.wrap(u0 + d)
            else:
                a = self.label(xs) - self.label(self.q_x[j])
                u1, u2 = a - d, a + d
            h += S[j] * reach * (np.exp(-u0 * u0 / s2)
                                 + edge * (np.exp(-u1 * u1 / s2) + np.exp(-u2 * u2 / s2)))
        # memoria interpolada entre filas
        f = rm - 0.5
        i0 = int(np.floor(f)) % n
        w1 = f - np.floor(f)
        mrow = (1 - w1) * self.mem[i0] + w1 * self.mem[(i0 + 1) % n]
        ci = np.rint(xs).astype(int) % n
        return self.A * h + self.beta * mrow[ci]

    def current(self, x, t):
        """Velocidad del liquido en (r = t, x).

        El objetivo pasa suavemente de la fila recien cruzada k a la siguiente
        k+1: E_c = (1-sig) E_k(c) + sig E_{k+1}(c), sig = smoothstep(p, 0.02, 0.3),
        con p la fraccion de camino desde el centro de k. Es continuo en t."""
        n = self.n
        k = int(np.floor(t - 0.5))
        p = t - (k + 0.5)
        sig = smoothstep(p, 0.02, 0.3)
        e = np.zeros(n)
        for row, wgt, lead in ((k % n, 1 - sig, x), ((k + 1) % n, sig, x + self.alpha * (1 - p))):
            if wgt < 1e-6:
                continue
            d = self.wrap(self.cols - lead)
            e += wgt * (self.heat(row + 0.5, self.cols, t, skip=row) + self.eps * d * d)
        e /= self.T
        w = np.exp(-(e - e.min()))
        Z = (w * self.phases).sum() / w.sum()
        phi = 2 * np.pi * x / n
        return self.kappa * abs(Z) * n / (2 * np.pi) * np.sin(np.angle(Z) - phi)

    # --- integracion -------------------------------------------------------
    def step(self):
        n, dt = self.n, self.dt
        u = self.current(self.x, self.t)
        self.v += dt * (u - self.v) / self.tau
        self.x = (self.x + dt * self.v) % n
        c_old = np.floor(self.t - 0.5)
        self.t += dt
        # memoria: pulso suave alrededor del centro de cada fila
        fr = (self.t % 1.0) - 0.5
        pulse = np.exp(-fr * fr / (2 * 0.15 ** 2)) / (0.15 * np.sqrt(2 * np.pi))
        row = int(np.floor(self.t % n))
        du = self.wrap(self.cols - self.x)
        share = np.exp(-du * du / (2 * 0.3 ** 2))
        self.mem *= np.exp(-self.lam * dt)
        self.mem[row] += dt * pulse * share / share.sum()
        if np.floor(self.t - 0.5) != c_old:     # centro de fila: nace una reina
            row = int(np.floor(self.t - 0.5)) % n
            self.q_t[row], self.q_x[row] = self.t, self.x
            self.lap_cols.append(int(np.rint(self.x)) % n)
            if row == n - 1:
                if len(self.lap_cols) == n:
                    self.laps.append(tuple(self.lap_cols))
                self.lap_cols = []

    def run(self, laps):
        target = len(self.laps) + laps
        while len(self.laps) < target:
            self.step()
        return self.laps
