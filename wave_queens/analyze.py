"""Aggregate a sweep over seeds, rank the cells, and show which knobs matter.

    ./venv/bin/python -m wave_queens.analyze std_A n_distinct
"""

import collections
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .sweep import load

R = "wave_queens/results"
AXES = ("temp", "tau", "sigma", "w_lat", "w_diag", "w_col", "bias_w",
        "n_replicas", "temp_min", "temp_max", "swap_every", "n_steps")


def group(rows, score):
    """Collapse seeds: one entry per parameter cell, with mean and spread."""
    cells = collections.defaultdict(list)
    for r in rows:
        key = tuple((a, r[a]) for a in AXES if a in r)
        cells[(r["n"], key)].append(r)
    out = []
    for (n, key), rs in cells.items():
        vals = np.array([r[score] for r in rs], float)
        out.append(dict(n=n, **dict(key), mean=vals.mean(), std=vals.std(),
                        best=vals.max(), seeds=len(rs),
                        perms=np.mean([r["n_perms"] for r in rs]),
                        flat=np.mean([r["flat_frac"] for r in rs]),
                        rate=np.mean([r.get("distinct_per_cpu_s", 0) for r in rs]),
                        total=rs[0]["n_solutions_total"]))
    return out


def marginals(cells, axes, score_label, path, title=""):
    """Mean score against each axis, marginalised over everything else.

    With a 5-D grid this is the honest first look: it says which knob moves the
    number and which is noise, without pretending a 2-D slice is the whole story.
    """
    ns = sorted({c["n"] for c in cells})
    fig, axs = plt.subplots(len(ns), len(axes), figsize=(3.0 * len(axes), 3.0 * len(ns)),
                            squeeze=False)
    for i, n in enumerate(ns):
        cs = [c for c in cells if c["n"] == n]
        for j, ax_name in enumerate(axes):
            vals = sorted({c[ax_name] for c in cs})
            m = [np.mean([c["mean"] for c in cs if c[ax_name] == v]) for v in vals]
            b = [np.max([c["mean"] for c in cs if c[ax_name] == v]) for v in vals]
            a = axs[i][j]
            a.plot(range(len(vals)), m, "o-", color="tab:blue", label="mean over cells")
            a.plot(range(len(vals)), b, "s--", color="tab:red", label="best cell")
            a.set_xticks(range(len(vals)), [f"{v:g}" for v in vals])
            a.set_xlabel(ax_name)
            if j == 0:
                a.set_ylabel(f"N={n}\n{score_label}")
            if i == 0 and j == len(axes) - 1:
                a.legend(fontsize=7)
    fig.suptitle(title or f"marginal effect on {score_label}")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def table(cells, k=12):
    axes = [a for a in AXES if a in cells[0]]
    for n in sorted({c["n"] for c in cells}):
        cs = sorted([c for c in cells if c["n"] == n],
                    key=lambda c: (-c["mean"], -c["best"]))[:k]
        print(f"\n--- N={n}   (of {cs[0]['total']} solutions, {cs[0]['seeds']} seeds/cell) ---")
        head = "".join(f"{a:>8}" for a in axes)
        print(f"{head}{'mean':>7}{'std':>6}{'best':>6}{'perms':>7}{'flat':>7}{'per_cpu_s':>11}")
        for c in cs:
            row = "".join(f"{c[a]:>8}" for a in axes)
            print(f"{row}{c['mean']:>7.2f}{c['std']:>6.2f}{c['best']:>6.0f}"
                  f"{c['perms']:>7.0f}{c['flat']:>7.3f}{c['rate']:>11.4f}")


if __name__ == "__main__":
    name = sys.argv[1]
    score = sys.argv[2] if len(sys.argv) > 2 else "n_distinct"
    rows = load(f"{R}/{name}.jsonl")
    cells = group(rows, score)
    print(f"{len(rows)} runs -> {len(cells)} parameter cells")
    table(cells)
    varying = [a for a in AXES if a in cells[0] and len({c[a] for c in cells}) > 1]
    marginals(cells, varying, score, f"{R}/marginals_{name}.png",
              title=f"{name}: which knob actually moves '{score}'  "
                    f"(blue = mean over cells, red = best cell)")
    print(f"\nwrote {R}/marginals_{name}.png")
