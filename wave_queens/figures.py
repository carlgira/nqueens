"""Render the featured runs.  ./venv/bin/python -m wave_queens.figures <name> [...]"""

import os
import sys

from .core import run
from .plots import run_figure, dwell_figure, phase_diagram
from .sweep import load

R = "wave_queens/results"

FEATURED = {}


def featured(name, cfg, window=None):
    FEATURED[name] = (cfg, window)


# modular: the temperature scan puts the intermittent band at T ~ 0.09-0.12.
featured("modular_n7", dict(mode="modular", n=7, temp=0.11, n_steps=500_000,
                            seed=2, lam_probe=1), (0, 600))
# standard: best cells of results/standard_grid.jsonl
featured("standard_n6", dict(mode="standard", n=6, temp=0.15, w_diag=1.0, w_lat=0.25,
                             n_steps=500_000, seed=1, lam_probe=1), (0, 1000))
featured("standard_n7", dict(mode="standard", n=7, temp=0.22, w_diag=0.5, w_lat=0.5,
                             n_steps=500_000, seed=1, lam_probe=1), (0, 1000))


def make(name):
    cfg, window = FEATURED[name]
    kpi, res = run(cfg, keep_traces=True)
    os.makedirs(R, exist_ok=True)
    run_figure(kpi, res, f"{R}/{name}.png", t_window=window)
    dwell_figure(kpi, res, f"{R}/{name}_dwell.png")
    keys = ("mode", "n", "temp", "V_min", "PR", "kin_per_queen", "lyapunov",
            "flat_frac", "n_configs", "n_perms", "n_visits", "n_distinct",
            "n_distinct_class", "n_solutions_total", "dwell_mean", "dwell_max",
            "coverage", "t_end")
    print(name, {k: (round(kpi[k], 4) if isinstance(kpi[k], float) else kpi[k])
                 for k in keys}, flush=True)
    return kpi


if __name__ == "__main__":
    args = sys.argv[1:] or list(FEATURED)
    for a in args:
        if a.startswith("phase:"):
            _, path, x, y, z = a.split(":")
            phase_diagram(load(f"{R}/{path}.jsonl"), x, y, z,
                          f"{R}/phase_{path}_{z}.png")
        else:
            make(a)
