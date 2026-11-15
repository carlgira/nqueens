"""Figures for the wave-queens experiments."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

TWO_PI = 2.0 * np.pi


def _cols(res, pot_mode, n):
    """Queen positions in column units.

    The modular potential has an exact zero mode: the whole splay lattice is
    free to rotate, and it does, steadily.  In the lab frame that drift smears
    every flat band into a diagonal stripe.  De-rotating by the lattice-locking
    phase arg(sum e^{iN theta})/N -- the same quantity the readout uses --
    removes only the zero mode and leaves the physics.
    """
    q = res["q"]
    if pot_mode != "modular":
        return q
    phi = np.angle(np.exp(1j * n * q).sum(axis=1)) / n
    return ((q - phi[:, None]) * n / TWO_PI) % n


def _profile(cols, n, kappa=0.16, pts=None):
    grid = np.linspace(0, n, pts or 10 * n, endpoint=False)
    d = np.abs(grid[None, :, None] - cols[:, None, :])
    d = np.minimum(d, n - d)
    return np.exp(-0.5 * (d / kappa) ** 2).sum(axis=2)


def _segments(x, y, jump):
    """Insert NaNs where a periodic coordinate wraps, so lines do not streak."""
    y = y.astype(float).copy()
    bad = np.abs(np.diff(y)) > jump
    y[1:][bad] = np.nan
    return x, y


def run_figure(kpi, res, path, t_window=None):
    """The four things we want to see: the superposition, the witness,
    the queens walking, and the itinerary through solution space."""
    n = kpi["n"]
    t = res["t"]
    lo, hi = t_window if t_window else (t[0], t[-1])
    sel = (t >= lo) & (t <= hi)
    visits = [v for v in res["visits"] if v[1] >= lo and v[0] <= hi]

    fig, ax = plt.subplots(4, 1, figsize=(13, 12), sharex=True,
                           gridspec_kw=dict(height_ratios=[1.1, 1, 1, 0.8]))
    fig.suptitle(
        f"wave queens · {kpi['mode']} · N={n} · T={kpi['temp']:.2f} · "
        f"lambda={kpi['lyapunov']:.2f} · PR={kpi['PR']:.3f} · "
        f"flat {100 * kpi['flat_frac']:.1f}% of the time · "
        f"{kpi['n_distinct']}/{kpi['n_solutions_total']} solutions visited",
        fontsize=12)

    cols_all = _cols(res, kpi["mode"], n)
    cols = cols_all[sel]

    # (A) the superposition itself: column occupancy profile over time.
    ax[0].imshow(_profile(cols, n).T, aspect="auto", origin="lower", cmap="magma",
                 extent=[lo, hi, 0, n], interpolation="nearest")
    ax[0].set_ylabel("column")
    ax[0].set_title("superposition  $\\sum_k \\kappa(c-x_k)$  — evenly spaced "
                    "horizontal bands = one queen per column", fontsize=10)

    # (B) the flatness witness.  Its floor is thermal, not zero: each soft mode
    # carries T/2 of potential energy, so V ~ (N/2) T even sitting in a solution.
    floor = 0.5 * n * kpi["temp"]
    ax[1].plot(t[sel], res["V"][sel] - kpi["v_star"], lw=0.5, color="0.25")
    ax[1].axhline(floor, color="tab:orange", lw=1, ls=":",
                  label=f"thermal floor $NT/2$ = {floor:.2f}")
    for v0, v1, _, _ in visits:
        ax[1].axvspan(v0, v1, color="tab:green", alpha=0.30, lw=0)
    ax[1].set_yscale("log")
    ax[1].set_ylabel("$V - V^*$")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].set_title("flatness witness — green = an exact solution is on the board",
                    fontsize=10)

    # (C) the queens walking
    for k in range(n):
        ax[2].plot(*_segments(t[sel], cols[:, k], 0.6 * n), lw=0.7)
    for v0, v1, _, _ in visits:
        ax[2].axvspan(v0, v1, color="tab:green", alpha=0.18, lw=0)
    ax[2].set_ylabel("column")
    ax[2].set_title("each queen's position — chaotic drift, then a rearrangement",
                    fontsize=10)

    # (D) itinerary: which solution, and how many distinct ones so far
    order, seen = {}, []
    for v in res["visits"]:
        if v[2] not in order:
            order[v[2]] = len(order)
        seen.append(v[0])
    cls_order = {}
    for v in res["visits"]:
        cls_order.setdefault(v[3], len(cls_order))
    for v0, v1, key, cls in visits:
        ax[3].plot([v0, v1], [order[key]] * 2, lw=6, solid_capstyle="butt",
                   color=plt.cm.tab10(cls_order[cls] % 10))
    ax[3].set_ylabel("solution #\n(order of discovery)")
    ax[3].set_xlabel("time")
    ax[3].set_title("itinerary — the orbit walks from one solution to another",
                    fontsize=10)
    ax[3].set_ylim(-1, max(len(order), 1))
    ax[3].set_xlim(lo, hi)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def dwell_figure(kpi, res, path):
    """Dwell-time statistics and the discovery curve."""
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    d = np.array([v[1] - v[0] for v in res["visits"]])
    if d.size:
        ax[0].hist(d, bins=np.logspace(np.log10(max(d.min(), 1e-3)),
                                       np.log10(d.max() * 1.05), 25), color="tab:green")
        ax[0].set_xscale("log")
    ax[0].set_xlabel("dwell time in a solution")
    ax[0].set_ylabel("count")
    ax[0].set_title(f"mean {kpi['dwell_mean']:.2f}, max {kpi['dwell_max']:.2f}")

    ts = [v[0] for v in res["visits"]]
    seen, curve = set(), []
    for v in res["visits"]:
        seen.add(v[2])
        curve.append(len(seen))
    ax[1].step(ts, curve, where="post", color="tab:blue")
    ax[1].axhline(kpi["n_solutions_total"], color="0.5", ls="--",
                  label=f"all {kpi['n_solutions_total']} solutions")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("distinct solutions found")
    ax[1].legend(fontsize=8)
    ax[1].set_title("discovery curve")

    v = res["V"] - kpi["v_star"]
    ax[2].hist(v, bins=120, color="0.35")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("$V - V^*$")
    ax[2].set_title("witness distribution — the spike at 0 is the flat phase")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def phase_diagram(rows, xkey, ykey, zkey, path, title=""):
    """Grid of runs -> heat map of a KPI, one panel per (mode, n)."""
    groups = {}
    for r in rows:
        groups.setdefault((r["mode"], r["n"]), []).append(r)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4.4),
                             squeeze=False)
    for ax, ((mode, n), rs) in zip(axes[0], sorted(groups.items())):
        xs = sorted({r[xkey] for r in rs})
        ys = sorted({r[ykey] for r in rs})
        Z = np.full((len(ys), len(xs)), np.nan)
        for r in rs:
            Z[ys.index(r[ykey]), xs.index(r[xkey])] = r[zkey]
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                       extent=[-0.5, len(xs) - 0.5, -0.5, len(ys) - 0.5])
        ax.set_xticks(range(len(xs)), [f"{v:g}" for v in xs])
        ax.set_yticks(range(len(ys)), [f"{v:g}" for v in ys])
        ax.set_xlabel(xkey)
        ax.set_ylabel(ykey)
        ax.set_title(f"{mode} N={n}")
        for i in range(len(ys)):
            for j in range(len(xs)):
                if not np.isnan(Z[i, j]):
                    ax.text(j, i, f"{Z[i, j]:.2f}".rstrip("0").rstrip("."),
                            ha="center", va="center", fontsize=7,
                            color="w" if Z[i, j] < np.nanmax(Z) * 0.6 else "k")
        fig.colorbar(im, ax=ax, label=zkey)
    fig.suptitle(title or zkey)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
