"""How the structural signatures scale with N.  Reads featured runs, never simulates.

    python -m wave_queens.scaling
"""

import collections

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .featured import load

R = "wave_queens/results"
NS = (7, 8, 9, 10)


def signatures(n):
    m = load(f"winner_n{n}")
    vis = m["visits"]
    d = np.array([v["t_end"] - v["t_start"] for v in vis])
    per = collections.defaultdict(list)
    for v in sorted(vis, key=lambda v: v["t_start"]):
        per[v["replica"]].append(v)
    hops = np.array([sum(x != y for x, y in zip(a["placement"], b["placement"]))
                     for vs in per.values() for a, b in zip(vs, vs[1:])
                     if a["placement"] != b["placement"]] or [np.nan])
    return m, d, hops


def figure(path=f"{R}/scaling_structure.png"):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = plt.cm.viridis(np.linspace(0, .85, len(NS)))

    for n, c in zip(NS, cols):
        m, d, hops = signatures(n)
        k = m["kpi"]
        lab = f"N={n} ({len(d)} visitas)"

        # 1. the dwell comb -- fine bins, no log, so the teeth are visible
        h, e = np.histogram(d, bins=np.arange(0, 3.0, 0.04))
        ax[0].plot(e[:-1] + .02, h / max(h.max(), 1), lw=1.3, color=c, label=lab)

        # 2. hop size as a fraction of N
        if not np.isnan(hops).all():
            hh, ee = np.histogram(hops / n, bins=np.linspace(0, 1.05, 22))
            ax[1].plot(ee[:-1] + .025, hh / max(hh.sum(), 1), "o-", ms=3,
                       lw=1.3, color=c, label=lab)

        # 3. yield vs the size of the target
        ax[2].scatter([k["n_classes_total"]], [k["n_distinct_class"]], s=90,
                      color=c, edgecolor="k", zorder=3, label=f"N={n}")

    for x in (0.38, 0.78, 1.18, 1.58, 2.38):
        ax[0].axvline(x, color="0.75", ls=":", lw=.8, zorder=0)
    ax[0].set_xlabel("tiempo de residencia")
    ax[0].set_ylabel("frecuencia (normalizada)")
    ax[0].legend(fontsize=8)
    ax[0].set_title("el peine de dwell no se mueve con N\n"
                    "(punteadas: 0.38, 0.78, 1.18, 1.58, 2.38)", fontsize=10)

    ax[1].axvline(0.88, color="k", ls="--", lw=1)
    ax[1].set_xlabel("reinas que se mueven / N")
    ax[1].set_ylabel("fraccion de transiciones")
    ax[1].legend(fontsize=8)
    ax[1].set_title("los saltos son globales a todo N\n"
                    "(discontinua: 0.88)", fontsize=10)

    tot = [load(f"winner_n{n}")["kpi"]["n_classes_total"] for n in NS]
    ax[2].plot(tot, tot, "--", color="0.6", label="cobertura total")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("familias que existen")
    ax[2].set_ylabel("familias encontradas")
    ax[2].legend(fontsize=8)
    ax[2].set_title("el rendimiento se queda plano\nmientras el blanco crece", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    figure()
