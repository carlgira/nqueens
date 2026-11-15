"""
Experimento de basin hopping: 10.000 universos en paralelo.
============================================================

En vez de afinar una sola simulacion, lanzamos miles de universos, cada uno
con su propia posicion inicial aleatoria y su propia energia cinetica aleatoria
(la idea del usuario: "individuos distintos", no clones). Dejamos que la fisica
del sistema disipativo evolucione y COSECHAMOS los universos que cruzan una
solucion valida. Los que se quedan atascados en un minimo local se desechan.

Toda la fisica esta vectorizada sobre el eje de universos (batch), asi que los
10.000 se integran a la vez con numpy. Es exactamente el mismo sistema de
newton_queens.py:  m_i r_i'' = -grad U_grid - sum_j k_j grad U_rep - gamma v.

Al final se imprimen estadisticas y se guarda una figura con la cosecha.
"""

from __future__ import annotations
import time, sys
import numpy as np

# ------------------------- configuracion del experimento --------------------
N          = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TRIALS     = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
MAX_STEPS  = int(sys.argv[3]) if len(sys.argv) > 3 else 6500
CHECK_EVERY = 5
CHUNK      = 2000          # universos por lote (memoria)
SEED       = 20260713

# constantes fisicas (config balistica: friccion baja -> exploracion larga)
A, B, SIGMA = 2.5, 8.0, 0.5
GAMMA       = float(sys.argv[4]) if len(sys.argv) > 4 else 0.03
V0          = 3.0         # escala de la energia cinetica inicial (aleatoria)
K_SPREAD    = 0.7         # heterogeneidad del poder individual k_i
MASS_COUPLE = 0.8
WALL_K, DT  = 30.0, 0.012
TAU         = 2 * np.pi
S2          = 2 * SIGMA ** 2


# ------------------------------ fisica vectorizada --------------------------
def forces(pos, vel, k, mass):
    """Fuerza sobre cada reina en cada universo. pos/vel: (Bt,N,2)."""
    x, y = pos[..., 0], pos[..., 1]                       # (Bt,N)
    fx = -TAU * A * np.sin(TAU * x)
    fy = -TAU * A * np.sin(TAU * y)

    dx = x[:, :, None] - x[:, None, :]                    # (Bt,N,N)
    dy = y[:, :, None] - y[:, None, :]
    u, v = dx - dy, dx + dy
    T1 = np.exp(-(dx * dx) / S2); T2 = np.exp(-(dy * dy) / S2)
    T3 = np.exp(-(u * u) / S2);   T4 = np.exp(-(v * v) / S2)
    kj = k[:, None, :] / SIGMA ** 2                       # poder de quien empuja
    fx += (kj * (dx * T1 + u * T3 + v * T4)).sum(axis=2)
    fy += (kj * (dy * T2 - u * T3 + v * T4)).sum(axis=2)

    lo, hi = 0.0, N - 1.0
    fx += WALL_K * np.maximum(lo - x, 0) - WALL_K * np.maximum(x - hi, 0)
    fy += WALL_K * np.maximum(lo - y, 0) - WALL_K * np.maximum(y - hi, 0)
    fx -= GAMMA * vel[..., 0]
    fy -= GAMMA * vel[..., 1]
    return np.stack([fx, fy], axis=-1)


def accel(pos, vel, k, mass):
    return forces(pos, vel, k, mass) / mass[..., None]


def solved_mask(pos):
    """True para cada universo cuya configuracion redondeada es valida."""
    c = np.clip(np.rint(pos), 0, N - 1).astype(np.int32)  # (Bt,N,2)
    cx, cy = c[..., 0], c[..., 1]
    same_col = cx[:, :, None] == cx[:, None, :]
    same_row = cy[:, :, None] == cy[:, None, :]
    same_dia = np.abs(cx[:, :, None] - cx[:, None, :]) == np.abs(cy[:, :, None] - cy[:, None, :])
    attack = same_col | same_row | same_dia
    eye = np.eye(N, dtype=bool)[None]
    attack &= ~eye
    conflicts = attack.any(axis=2).sum(axis=1)            # pares atacando (por reina)
    return conflicts == 0, c


def run_chunk(bt, rng):
    """Integra `bt` universos; devuelve (solved_step, boards, final_conflicts)."""
    pos = rng.uniform(0.0, N - 1.0, size=(bt, N, 2))
    vel = rng.normal(0.0, V0, size=(bt, N, 2))
    k = B * (1.0 + K_SPREAD * rng.uniform(-1.0, 1.0, size=(bt, N)))
    k = np.clip(k, 0.05 * B, None)
    mass = 1.0 - MASS_COUPLE + MASS_COUPLE * k / k.mean(axis=1, keepdims=True)

    solved_step = np.full(bt, -1, dtype=np.int32)
    boards = np.zeros((bt, N, 2), dtype=np.int32)
    active = np.ones(bt, dtype=bool)                      # aun no resueltos

    for step in range(MAX_STEPS):
        # RK4 (solo se aplica a universos activos; los resueltos se congelan)
        a1 = accel(pos, vel, k, mass)
        p2, v2 = pos + 0.5 * DT * vel, vel + 0.5 * DT * a1
        a2 = accel(p2, v2, k, mass)
        p3, v3 = pos + 0.5 * DT * v2, vel + 0.5 * DT * a2
        a3 = accel(p3, v3, k, mass)
        p4, v4 = pos + DT * v3, vel + DT * a3
        a4 = accel(p4, v4, k, mass)
        dpos = (DT / 6) * (vel + 2 * v2 + 2 * v3 + v4)
        dvel = (DT / 6) * (a1 + 2 * a2 + 2 * a3 + a4)
        m = active[:, None, None]
        pos = pos + dpos * m
        vel = vel + dvel * m

        if step % CHECK_EVERY == 0 or step == MAX_STEPS - 1:
            sol, cells = solved_mask(pos)
            fresh = sol & active
            if fresh.any():
                solved_step[fresh] = step
                boards[fresh] = cells[fresh]
                active[fresh] = False
            if not active.any():
                break

    # conflictos finales de los que quedaron atascados
    final_conf = conflict_count(pos)
    return solved_step, boards, final_conf


def conflict_count(pos):
    c = np.clip(np.rint(pos), 0, N - 1).astype(np.int32)
    cx, cy = c[..., 0], c[..., 1]
    same_col = cx[:, :, None] == cx[:, None, :]
    same_row = cy[:, :, None] == cy[:, None, :]
    same_dia = np.abs(cx[:, :, None] - cx[:, None, :]) == np.abs(cy[:, :, None] - cy[:, None, :])
    attack = (same_col | same_row | same_dia) & ~np.eye(N, dtype=bool)[None]
    return attack.sum(axis=(1, 2)) // 2


# ------------------------------ simetrias / canonico ------------------------
def canonical(cells):
    """Forma canonica de una solucion (minima sobre las 8 simetrias del cuadrado, D4)."""
    n = N
    pts = [tuple(map(int, p)) for p in cells]
    forms = [
        lambda x, y: (x, y),                 # identidad
        lambda x, y: (n - 1 - x, y),         # reflexion vertical
        lambda x, y: (x, n - 1 - y),         # reflexion horizontal
        lambda x, y: (n - 1 - x, n - 1 - y), # rotacion 180
        lambda x, y: (y, x),                 # reflexion diagonal
        lambda x, y: (y, n - 1 - x),         # rotacion 90
        lambda x, y: (n - 1 - y, x),         # rotacion 270
        lambda x, y: (n - 1 - y, n - 1 - x), # reflexion antidiagonal
    ]
    return min(tuple(sorted(f(x, y) for x, y in pts)) for f in forms)


# ---------------------------------- main ------------------------------------
def main():
    print(f"# Basin hopping  N={N}  intentos={TRIALS}  max_steps={MAX_STEPS}")
    print(f"# fisica: A={A} B={B} sigma={SIGMA} gamma={GAMMA} v0={V0} k_spread={K_SPREAD}")
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    solved_steps, fail_conf = [], []
    solution_cells = []
    done = 0
    while done < TRIALS:
        bt = min(CHUNK, TRIALS - done)
        ss, boards, fc = run_chunk(bt, rng)
        for i in range(bt):
            if ss[i] >= 0:
                solved_steps.append(int(ss[i]))
                solution_cells.append(boards[i].copy())
            else:
                fail_conf.append(int(fc[i]))
        done += bt
        print(f"  {done:6d}/{TRIALS}   resueltos hasta ahora: {len(solved_steps):5d}"
              f"   ({time.time()-t0:5.1f}s)", flush=True)

    n_ok = len(solved_steps)
    print("\n" + "=" * 58)
    print(f"RESULTADOS  ({time.time()-t0:.1f}s)")
    print("=" * 58)
    print(f"intentos totales      : {TRIALS}")
    print(f"soluciones validas    : {n_ok}   ({100*n_ok/TRIALS:.2f}%)")
    print(f"atascados (desechados): {TRIALS-n_ok}   ({100*(TRIALS-n_ok)/TRIALS:.2f}%)")

    if solved_steps:
        ss = np.array(solved_steps)
        print(f"\npasos hasta la solucion (de los que resolvieron):")
        print(f"  min={ss.min()}  p25={int(np.percentile(ss,25))}  "
              f"mediana={int(np.median(ss))}  media={ss.mean():.0f}  "
              f"p90={int(np.percentile(ss,90))}  max={ss.max()}")

    # cuantas soluciones DISTINTAS descubrimos
    exact = set()
    fundamental = set()
    for cells in solution_cells:
        exact.add(tuple(sorted(map(tuple, cells.tolist()))))
        fundamental.add(canonical(cells))
    print(f"\ndiversidad de la cosecha:")
    print(f"  soluciones exactas distintas   : {len(exact)}")
    print(f"  soluciones fundamentales (unicas por simetria): {len(fundamental)}")

    # guardar las soluciones exactas distintas a disco
    with open(f"solutions_N{N}.txt", "w") as fh:
        fh.write(f"# {len(exact)} soluciones exactas distintas de N={N} "
                 f"cosechadas de {TRIALS} universos\n")
        for sol in sorted(exact):
            fh.write(" ".join(f"({x},{y})" for x, y in sol) + "\n")
    print(f"  soluciones guardadas en solutions_N{N}.txt")
    known = {4:2,5:10,6:4,7:40,8:92,9:352,10:724,11:2680,12:14200}
    knownf = {4:1,5:2,6:1,7:6,8:12,9:46,10:92,11:341,12:1787}
    if N in known:
        print(f"  total real para N={N}: {known[N]} soluciones "
              f"({knownf[N]} fundamentales) -> cubrimos "
              f"{100*len(exact)/known[N]:.1f}% / {100*len(fundamental)/knownf[N]:.1f}%")

    if fail_conf:
        fc = np.array(fail_conf)
        vals, cnts = np.unique(fc, return_counts=True)
        print(f"\nconflictos finales de los atascados (donde quedaron paralizados):")
        for v, c in zip(vals[:10], cnts[:10]):
            bar = "#" * int(40 * c / cnts.max())
            print(f"  {v:2d} conflictos : {c:5d}  {bar}")

    plot(np.array(solved_steps), np.array(fail_conf), n_ok, exact, fundamental)


def plot(solved_steps, fail_conf, n_ok, exact, fundamental):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("(sin matplotlib, no se genera figura)", e)
        return

    plt.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#0d1117",
                         "text.color": "#e6edf6", "axes.labelcolor": "#8ea0b6",
                         "xtick.color": "#8ea0b6", "ytick.color": "#8ea0b6",
                         "axes.edgecolor": "#30363d", "font.size": 9})
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle(f"Newton Queens · basin hopping N={N} · {TRIALS} universos aleatorios",
                 color="#4fe0cf", fontsize=13, y=1.02)

    ax[0].bar(["cosechadas", "atascadas"], [n_ok, TRIALS - n_ok],
              color=["#4fe0cf", "#ff556e"])
    ax[0].set_title(f"{n_ok} soluciones de {TRIALS}  ({100*n_ok/TRIALS:.1f}%)")
    for i, v in enumerate([n_ok, TRIALS - n_ok]):
        ax[0].text(i, v, f"{v}", ha="center", va="bottom", color="#e6edf6")

    if len(solved_steps):
        ax[1].hist(solved_steps, bins=40, color="#7b6cff", alpha=0.85)
        ax[1].axvline(np.median(solved_steps), color="#ffb454", ls="--",
                      label=f"mediana {int(np.median(solved_steps))}")
        ax[1].legend(facecolor="#131c28", edgecolor="#30363d", labelcolor="#e6edf6")
    ax[1].set_title("pasos hasta cristalizar")
    ax[1].set_xlabel("pasos de integracion")

    if len(fail_conf):
        vals, cnts = np.unique(fail_conf, return_counts=True)
        ax[2].bar(vals, cnts, color="#ff556e", alpha=0.8)
    ax[2].set_title("conflictos de los atascados")
    ax[2].set_xlabel("nº de conflictos al congelarse")

    plt.tight_layout()
    out = f"experiment_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigura guardada en {out}")

    # tablero(s) de ejemplo de la cosecha
    if exact:
        examples = list(exact)[:6]
        fig2, axs = plt.subplots(1, len(examples), figsize=(2.4 * len(examples), 2.6))
        if len(examples) == 1: axs = [axs]
        for a, sol in zip(axs, examples):
            grid = np.indices((N, N)).sum(0) % 2
            a.imshow(grid, cmap="bone", alpha=0.25)
            xs = [p[0] for p in sol]; ys = [p[1] for p in sol]
            a.scatter(xs, ys, s=120, c="#4fe0cf", edgecolors="white", zorder=3)
            a.set_xticks([]); a.set_yticks([]); a.set_xlim(-.5, N-.5); a.set_ylim(-.5, N-.5)
        fig2.suptitle("muestras distintas de la cosecha", color="#4fe0cf")
        plt.tight_layout()
        fig2.savefig(f"experiment_N{N}_samples.png", dpi=130, bbox_inches="tight")
        print(f"muestras guardadas en experiment_N{N}_samples.png")


if __name__ == "__main__":
    main()
