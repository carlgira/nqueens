"""
Newton Queens
=============

Las N-Reinas resueltas como un sistema fisico continuo, caotico y disipativo.

En vez de buscar en un grafo discreto, soltamos N particulas en un plano 2D
y las dejamos evolucionar bajo F = m a. Las reglas del ajedrez se codifican
como campos de energia potencial continuos:

  1. Tablero  ->  U_grid : un "carton de huevos" que atrae cada particula
                  hacia coordenadas enteras (las casillas validas).

        U_grid(x, y) = -A ( cos(2 pi x) + cos(2 pi y) )

  2. Reinas   ->  U_rep : repulsion ANISOTROPICA en forma de estrella. Solo
                  se dispara cuando dos reinas se alinean en fila, columna
                  o alguna de las dos diagonales.

        U_rep(i,j) = B [ g(dx) + g(dy) + g(dx-dy) + g(dx+dy) ],
        con dx = xi-xj, dy = yi-yj, g(s) = exp(-s^2 / 2 sigma^2)

  3. Disipacion -> friccion viscosa -gamma * v, que roba energia cinetica
                   hasta que el caos colapsa en un punto de reposo.

La ecuacion de movimiento acoplada para cada reina i es:

    m_i r_i'' = -grad U_grid(r_i)  -  sum_{j!=i} k_j grad U_rep(i,j)  -  gamma r_i'

HETEROGENEIDAD (romper la simetria en vez de meter temperatura)
---------------------------------------------------------------
El sistema homogeneo de N clones identicos se queda atascado: dos reinas
perfectamente alineadas forman un punto de fuerza nula (el gradiente de una
gaussiana se anula en su cima), asi que nada las separa. La cura fisica es
romper la homogeneidad y dar a cada reina su propia identidad:

  * Energia cinetica inicial aleatoria (v0): cada reina arranca con inercia
    propia. Una que caiga en un minimo local falso lleva impulso suficiente
    para subir la colina y escapar, como un pendulo con velocidad de sobra.

  * Coeficiente de poder individual k_i: en vez de una unica constante B,
    cada reina proyecta un campo de fuerza propio. Una reina "poderosa"
    (k_i alto) empuja con violencia; una "debil" (k_i bajo) apenas afecta a
    las demas. La fuerza que recibe la reina i viene ponderada por el poder
    k_j de quien la empuja.

  * Masa acoplada al poder (m_i ~ k_i): las poderosas son pesadas e
    (casi) inamovibles; las debiles son ligeras y salen despedidas. Se
    establece una jerarquia -- los cuerpos poderosos reclaman sus dimensiones
    y barren a los debiles hacia las zonas vacias, acelerando la formacion
    del patron. Es caos, pero un caos jerarquico.

El estado de reposo de este sistema disipativo (todas las fuerzas anuladas,
cada particula en un pozo, sin conflictos) ES una solucion al problema.
Se integra con Runge-Kutta 4 sobre el estado completo (posiciones + velocidades).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# Parametros del sistema fisico
# ---------------------------------------------------------------------------
@dataclass
class Params:
    N: int = 8            # numero de reinas / tamano del tablero
    A: float = 2.5        # profundidad de los pozos del tablero (gravedad local)
    B: float = 8.0        # poder de repulsion medio entre reinas
    sigma: float = 0.5    # ancho del "rayo laser" de ataque de la reina
    gamma: float = 0.03   # friccion baja: las reinas vuelan y exploran mucho tiempo
    mass: float = 1.0     # masa base (se reescala por el poder de cada reina)
    wall_k: float = 30.0  # rigidez de las paredes blandas del tablero
    dt: float = 0.012     # paso de integracion

    # --- Heterogeneidad: romper la simetria del sistema ---
    v0: float = 3.2       # escala de la velocidad cinetica inicial (inercia)
    k_spread: float = 0.7 # dispersion del poder individual k_i (0 = clones)
    mass_couple: float = 0.8  # cuanto la masa sigue al poder (0..1)

    # Recocido termico OPCIONAL (apagado por defecto: T0=0)
    T0: float = 0.0       # temperatura inicial (0 = sin ruido termico)
    cooling: float = 0.9975  # factor de enfriamiento por paso (T *= cooling)
    T_min: float = 1e-4   # temperatura minima


class NewtonQueens:
    """Motor de simulacion fisica de las N-Reinas."""

    def __init__(self, params: Params | None = None, seed: int | None = None):
        self.p = params or Params()
        self.rng = np.random.default_rng(seed)
        self.T = self.p.T0
        self.steps = 0
        self.reset_random()

    # ------------------------------------------------------------------ setup
    def reset_random(self):
        """Suelta las N reinas: posiciones y velocidades aleatorias, poderes heterogeneos."""
        p = self.p
        N = p.N
        # posiciones aleatorias dentro del tablero [0, N-1]
        self.pos = self.rng.uniform(0.0, N - 1.0, size=(N, 2))
        # energia cinetica inicial: cada reina arranca con inercia propia
        self.vel = self.rng.normal(0.0, p.v0, size=(N, 2))
        # poder individual k_i (heterogeneo): unas reinas empujan mas que otras
        self.k = p.B * (1.0 + p.k_spread * self.rng.uniform(-1.0, 1.0, size=N))
        self.k = np.clip(self.k, 0.05 * p.B, None)
        # masa acoplada al poder: las poderosas son pesadas e inamovibles
        self.mass = p.mass * (1.0 - p.mass_couple + p.mass_couple * self.k / self.k.mean())
        self.T = p.T0
        self.steps = 0
        return self

    # --------------------------------------------------------------- energias
    def energy(self, pos: np.ndarray | None = None) -> dict:
        """Descompone la energia potencial total del estado actual."""
        p = self.p
        if pos is None:
            pos = self.pos
        x, y = pos[:, 0], pos[:, 1]

        u_grid = -p.A * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)).sum()

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        s2 = 2 * p.sigma ** 2
        g = lambda s: np.exp(-(s ** 2) / s2)
        # solo pares i<j
        iu = np.triu_indices(p.N, k=1)
        u_rep = p.B * (g(dx) + g(dy) + g(dx - dy) + g(dx + dy))[iu].sum()

        kinetic = 0.5 * (self.mass[:, None] * self.vel ** 2).sum()
        return dict(grid=u_grid, rep=u_rep, potential=u_grid + u_rep,
                    kinetic=kinetic, total=u_grid + u_rep + kinetic)

    # ----------------------------------------------------------------- fuerzas
    def forces(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Fuerza total sobre cada reina: -grad U_grid - grad U_rep - friccion - paredes."""
        p = self.p
        N = p.N
        x, y = pos[:, 0], pos[:, 1]

        # --- fuerza del tablero (carton de huevos) ---
        #   -dU_grid/dx = -2 pi A sin(2 pi x)  (restaura hacia el entero mas cercano)
        fx = -2 * np.pi * p.A * np.sin(2 * np.pi * x)
        fy = -2 * np.pi * p.A * np.sin(2 * np.pi * y)

        # --- repulsion anisotropica entre pares de reinas ---
        dx = x[:, None] - x[None, :]     # dx[i,j] = xi - xj
        dy = y[:, None] - y[None, :]
        u = dx - dy                       # diagonal /
        v = dx + dy                       # diagonal \
        s2 = 2 * p.sigma ** 2
        T1 = np.exp(-(dx ** 2) / s2)
        T2 = np.exp(-(dy ** 2) / s2)
        T3 = np.exp(-(u ** 2) / s2)
        T4 = np.exp(-(v ** 2) / s2)

        # cada columna j se pondera por el PODER k_j de quien empuja:
        # una reina poderosa proyecta un campo intenso sobre las demas.
        kj = self.k[None, :] / p.sigma ** 2
        frx = (kj * (dx * T1 + u * T3 + v * T4)).sum(axis=1)
        fry = (kj * (dy * T2 - u * T3 + v * T4)).sum(axis=1)
        fx += frx
        fy += fry

        # --- paredes blandas para mantener las reinas dentro de [0, N-1] ---
        lo, hi = 0.0, N - 1.0
        fx += p.wall_k * np.maximum(lo - x, 0.0) - p.wall_k * np.maximum(x - hi, 0.0)
        fy += p.wall_k * np.maximum(lo - y, 0.0) - p.wall_k * np.maximum(y - hi, 0.0)

        # --- friccion viscosa (disipacion) ---
        fx -= p.gamma * vel[:, 0]
        fy -= p.gamma * vel[:, 1]

        return np.stack([fx, fy], axis=1)

    # ------------------------------------------------------------- integrador
    def _deriv(self, pos, vel):
        # aceleracion = fuerza / masa individual (las poderosas apenas se mueven)
        acc = self.forces(pos, vel) / self.mass[:, None]
        return vel, acc

    def step(self):
        """Un paso de Runge-Kutta 4 sobre (posiciones, velocidades)."""
        dt = self.p.dt
        pos, vel = self.pos, self.vel

        k1p, k1v = self._deriv(pos, vel)
        k2p, k2v = self._deriv(pos + 0.5 * dt * k1p, vel + 0.5 * dt * k1v)
        k3p, k3v = self._deriv(pos + 0.5 * dt * k2p, vel + 0.5 * dt * k2v)
        k4p, k4v = self._deriv(pos + dt * k3p, vel + dt * k3v)

        self.pos = pos + (dt / 6.0) * (k1p + 2 * k2p + 2 * k3p + k4p)
        self.vel = vel + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)

        # --- ruido termico (recocido): patadas aleatorias que se enfrian ---
        if self.T > self.p.T_min:
            self.vel += self.rng.normal(0.0, self.T, size=self.vel.shape) * np.sqrt(dt)
            self.T *= self.p.cooling

        self.steps += 1
        return self

    # ------------------------------------------------------------- solucion
    def board(self) -> np.ndarray:
        """Redondea las posiciones actuales a la casilla mas cercana."""
        return np.clip(np.rint(self.pos), 0, self.p.N - 1).astype(int)

    def conflicts(self, board: np.ndarray | None = None) -> int:
        """Cuenta pares de reinas que se atacan en la configuracion discreta."""
        if board is None:
            board = self.board()
        N = self.p.N
        c = 0
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi = board[i]
                xj, yj = board[j]
                if xi == xj or yi == yj or abs(xi - xj) == abs(yi - yj):
                    c += 1
        return c

    def is_solved(self) -> bool:
        """True si las reinas ocupan casillas unicas y no hay conflictos."""
        b = self.board()
        unique = len({tuple(r) for r in b}) == self.p.N
        return unique and self.conflicts(b) == 0

    def run(self, max_steps: int = 8000, settle_speed: float = 1e-3,
            stop_on_solution: bool = True, verbose: bool = False):
        """Integra hasta que el sistema se detiene o encuentra una solucion.

        Con stop_on_solution, congelamos el instante en que la configuracion
        redondeada ya es valida: el sistema en movimiento cruza el atractor y
        lo atrapamos alli (la friccion luego lo asentaria en ese mismo pozo).
        """
        for _ in range(max_steps):
            self.step()
            if stop_on_solution and self.is_solved():
                break
            speed = np.sqrt((self.vel ** 2).sum(axis=1)).max()
            if self.T <= self.p.T_min and speed < settle_speed:
                break
            if verbose and self.steps % 500 == 0:
                e = self.energy()
                print(f"step {self.steps:5d}  Ekin={e['kinetic']:7.2f}  "
                      f"Urep={e['rep']:8.3f}  conflicts={self.conflicts()}")
        return self


def solve(N: int = 8, seed: int | None = None, restarts: int = 400,
          max_steps: int = 6500, verbose: bool = False, **kw
          ) -> tuple[NewtonQueens, bool]:
    """Basin hopping: reinicia desde universos aleatorios hasta cosechar una solucion."""
    for attempt in range(restarts):
        sim = NewtonQueens(Params(N=N, **kw),
                           seed=None if seed is None else seed + attempt)
        sim.run(max_steps=max_steps)
        if sim.is_solved():
            if verbose:
                print(f"Resuelto en el intento {attempt + 1}, {sim.steps} pasos")
            return sim, True
    return sim, False


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sim, ok = solve(N=N, seed=1, verbose=True)
    b = sim.board()
    print("\nSolucion valida:" if ok else "\nMejor intento (no valido):")
    grid = [["." for _ in range(N)] for _ in range(N)]
    for x, y in b:
        grid[y][x] = "Q"
    print("\n".join(" ".join(row) for row in grid))
    print(f"\nconflictos = {sim.conflicts()}   pasos = {sim.steps}")
