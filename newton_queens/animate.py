"""
Visualizacion en vivo del colapso caotico de las Newton Queens.

Muestra el tablero como un mapa de calor del potencial U_grid (el carton de
huevos), las reinas como particulas brillantes, y las lineas de ataque que
parpadean en rojo mientras hay conflictos y se apagan al converger.

Uso:
    python animate.py [N]                 # ventana interactiva
    python animate.py [N] --gif salida.gif
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from newton_queens import NewtonQueens, Params


def attack_pairs(board):
    """Devuelve los pares de reinas (indices) que se atacan."""
    N = len(board)
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi = board[i]
            xj, yj = board[j]
            if xi == xj or yi == yj or abs(xi - xj) == abs(yi - yj):
                pairs.append((i, j))
    return pairs


def run(N=8, gif=None, steps_per_frame=4, max_frames=1500, seed=7):
    sim = NewtonQueens(Params(N=N), seed=seed)

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # fondo: potencial del tablero (carton de huevos)
    g = 400
    xs = np.linspace(-0.5, N - 0.5, g)
    X, Y = np.meshgrid(xs, xs)
    U = -sim.p.A * (np.cos(2 * np.pi * X) + np.cos(2 * np.pi * Y))
    ax.imshow(U, extent=[-0.5, N - 0.5, -0.5, N - 0.5], origin="lower",
              cmap="magma", alpha=0.55, interpolation="bilinear")

    (queens,) = ax.plot([], [], "o", ms=20, color="#58e6d9",
                        mec="white", mew=1.4, zorder=5)
    (ghost,) = ax.plot([], [], "s", ms=22, mfc="none", mec="#58e6d9",
                       alpha=0.35, zorder=4)  # casilla objetivo redondeada
    lines = []
    title = ax.set_title("", color="white", fontsize=13, pad=12)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.grid(True, color="white", alpha=0.12)
    ax.tick_params(colors="#8b949e")
    for s in ax.spines.values():
        s.set_color("#30363d")

    # basin hopping en vivo: universo tras universo hasta que uno cristaliza
    state = {"done_at": None, "attempt": 1, "stuck": 0}

    def update(frame):
        nonlocal lines
        if state["done_at"] is None:
            for _ in range(steps_per_frame):
                sim.step()

        pos = sim.pos
        queens.set_data(pos[:, 0], pos[:, 1])
        b = sim.board()
        ghost.set_data(b[:, 0], b[:, 1])

        for ln in lines:
            ln.remove()
        lines = []
        conf = attack_pairs(b)
        for i, j in conf:
            ln, = ax.plot([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]],
                          "-", color="#ff5555", alpha=0.55, lw=1.6, zorder=3)
            lines.append(ln)

        solved = sim.is_solved()
        if solved and state["done_at"] is None:
            state["done_at"] = frame
            queens.set_color("#7ee787")

        # este universo se paralizo sin resolver -> desecharlo y probar otro
        speed = (sim.vel ** 2).sum(axis=1).max() ** 0.5
        if state["done_at"] is None and speed < 3e-3:
            state["stuck"] += 1
            if state["stuck"] > 8:
                sim.reset_random()
                state["attempt"] += 1
                state["stuck"] = 0
        else:
            state["stuck"] = 0

        e = sim.energy()
        status = "CRISTALIZADO" if solved else f"{len(conf)} conflictos"
        title.set_text(
            f"N={N}   universo #{state['attempt']}   paso {sim.steps}   "
            f"E_cin={e['kinetic']:.1f}   {status}")

        if state["done_at"] is not None and frame - state["done_at"] > 60:
            anim.event_source.stop()
        return queens, ghost, title

    anim = FuncAnimation(fig, update, frames=max_frames,
                         interval=20, blit=False, repeat=False)

    if gif:
        from matplotlib.animation import PillowWriter
        print(f"Renderizando {gif} ...")
        anim.save(gif, writer=PillowWriter(fps=30), dpi=90)
        print("Listo.")
    else:
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    N = 8
    gif = None
    if args and args[0].isdigit():
        N = int(args[0])
    if "--gif" in args:
        gif = args[args.index("--gif") + 1]
    run(N=N, gif=gif)
