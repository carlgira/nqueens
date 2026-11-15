"""Featured runs: config + KPIs + traces, saved once, re-read instantly.

The stage-C sweep only stored counts, so "were those solutions modular?" could
not be answered from disk.  A featured run saves everything a later question
might need -- the config, every KPI, the full solution list, the visit itinerary
and the trajectory -- keyed by run_id.  Re-asking never re-simulates.

    python -m wave_queens.featured run   winner_n7    # simulate + store
    python -m wave_queens.featured show  winner_n7    # read back, no sim
"""

import json
import os
import sys
import time

import numpy as np

from .core import DEFAULTS, enumerate_solutions, run, run_id, run_pt

STORE = "wave_queens/results/featured"

# The stage-C winner: best cell for N=7 (10.5 distinct averaged over seeds, 12 on
# seed 2) and also the best distinct_per_cpu_s in its table.  Longer than the
# sweep version so the itinerary has enough transitions to look for structure in.
RUNS = {
    "winner_n7": dict(
        mode="standard", n=7, n_replicas=16, temp_min=0.12, temp_max=1.6,
        swap_every=100, tau=2.0, sigma=0.40, w_lat=0.15, w_diag=1.0,
        n_steps=300_000, seed=2),

    # best cell of scale_8910 for N=8; run long, with traces, to measure how the
    # structural signatures (dwell comb, hop size, family sweep) scale with N.
    "winner_n8": dict(
        mode="standard", n=8, col_norm="pairs", n_replicas=20, temp_min=0.12,
        temp_max=1.6, swap_every=100, tau=2.0, sigma=0.6, w_lat=0.15,
        w_diag=1.0, w_col=4.8, n_steps=300_000, seed=2),

    # best cell of scale_8910 for N=9; run long, with traces, to measure how the
    # structural signatures (dwell comb, hop size, family sweep) scale with N.
    "winner_n9": dict(
        mode="standard", n=9, col_norm="pairs", n_replicas=20, temp_min=0.25,
        temp_max=1.6, swap_every=100, tau=2.0, sigma=0.6, w_lat=0.15,
        w_diag=1.0, w_col=4.8, n_steps=300_000, seed=2),

    # best cell of scale_8910 for N=10; run long, with traces, to measure how the
    # structural signatures (dwell comb, hop size, family sweep) scale with N.
    "winner_n10": dict(
        mode="standard", n=10, col_norm="pairs", n_replicas=20, temp_min=0.12,
        temp_max=1.6, swap_every=100, tau=2.0, sigma=0.6, w_lat=0.15,
        w_diag=1.0, w_col=9.6, n_steps=300_000, seed=2),

    # N=11 scouting.  Everything intensive carries over unchanged (V per queen
    # is 0.70 flat in N, because the Nose-Hoover chain is per-queen, so the
    # temperatures transfer).  Two things were predicted to scale:
    #   w_col     = N/7 = 1.6  -- the column term is normalised by 1/N^2, so it
    #                            weakens as 1/N against the diagonals
    #   n_replicas= 16*sqrt(N/7) = 20 -- V fluctuations grow as sqrt(N), so the
    #                            ladder needs more rungs to hold swap_rate ~0.55
    # Target is no longer coverage: N=11 has 2680 solutions in 341 classes.
    # This run exists to measure flat_frac and the recrystallisation rate.
    "scout_n11": dict(
        mode="standard", n=11, n_replicas=20, temp_min=0.12, temp_max=1.6,
        swap_every=100, tau=2.0, sigma=0.40, w_lat=0.15, w_diag=1.0, w_col=1.6,
        n_steps=300_000, seed=2),

    # Control: identical but with w_col left at the N=7 value.  If the 1/N
    # argument is right this should place more queens in the same column and
    # find fewer solutions.
    "scout_n11_wcol1": dict(
        mode="standard", n=11, n_replicas=20, temp_min=0.12, temp_max=1.6,
        swap_every=100, tau=2.0, sigma=0.40, w_lat=0.15, w_diag=1.0, w_col=1.0,
        n_steps=300_000, seed=2),
}


def path_for(name):
    return f"{STORE}/{name}.json", f"{STORE}/{name}_traces.npz"


def do_run(name):
    cfg = RUNS[name]
    full = dict(DEFAULTS)
    full.update(cfg)
    os.makedirs(STORE, exist_ok=True)
    t0 = time.time()
    kpi, res = (run_pt(full, keep_traces=True) if full.get("n_replicas", 1) > 1
                else run(full, keep_traces=True))
    kpi["wall_s"] = time.time() - t0

    modular = {tuple(s) for s in enumerate_solutions(full["n"], modular=True)}
    visits = [dict(t_start=v[0], t_end=v[1], placement=list(v[2]),
                   klass=list(v[3]), replica=v[4] if len(v) > 4 else 0,
                   modular=tuple(v[2]) in modular) for v in res["visits"]]
    kpi["solutions_modular"] = sorted(list(s) for s in
                                      {tuple(v["placement"]) for v in visits}
                                      & modular)
    kpi["solutions_non_modular"] = sorted(list(s) for s in
                                          {tuple(v["placement"]) for v in visits}
                                          - modular)

    jpath, npath = path_for(name)
    with open(jpath, "w") as f:
        json.dump(dict(name=name, config=full, kpi=kpi, visits=visits), f, indent=1)
    np.savez_compressed(npath, t=res["t"], V=res["V"], q=res["q"],
                        profile=res["profile"], grid=res["grid"],
                        temps=res.get("temps", []))
    print(f"stored {jpath} ({kpi['wall_s']:.0f}s)  and {npath}")
    return load(name)


def load(name):
    jpath, npath = path_for(name)
    with open(jpath) as f:
        meta = json.load(f)
    meta["traces"] = dict(np.load(npath))
    return meta


if __name__ == "__main__":
    what, name = sys.argv[1], sys.argv[2]
    m = do_run(name) if what == "run" else load(name)
    k = m["kpi"]
    print(f"\n{name}: {k['n_distinct']}/{k['n_solutions_total']} solutions, "
          f"{k['n_distinct_class']}/{k['n_classes_total']} classes, "
          f"{len(m['visits'])} visits, flat_frac(cold)={k['flat_frac']:.3f}")
    print(f"  modular:     {len(k['solutions_modular'])}")
    print(f"  non-modular: {len(k['solutions_non_modular'])}")
