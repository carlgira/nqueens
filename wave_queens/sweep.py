"""Experiment harness: config grid -> one JSONL row of KPIs per run.

Append-only, resumable, parallel.  The run id is a hash of the config, so
re-launching a grid skips whatever is already on disk and only fills the holes.

    ./venv/bin/python -m wave_queens.sweep grids.py:coarse_standard results/coarse.jsonl
"""

import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

from .core import DEFAULTS, run, run_id, run_pt


def expand(grid):
    """{'temp': [.1,.2], 'n': [6,7]} -> list of configs (cartesian product)."""
    keys = list(grid)
    vals = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


def _done_ids(path):
    if not os.path.exists(path):
        return set()
    ids = set()
    with open(path) as f:
        for line in f:
            try:
                ids.add(json.loads(line)["run_id"])
            except Exception:
                pass
    return ids


def _full(cfg):
    c = dict(DEFAULTS)
    c.update(cfg)
    return c


def _worker(cfg):
    t0 = time.time()
    kpi, _ = (run_pt(cfg) if cfg.get("n_replicas", 1) > 1 else run(cfg))
    kpi["wall_s"] = time.time() - t0
    kpi["distinct_per_cpu_s"] = kpi["n_distinct"] / kpi["wall_s"]
    return kpi


def sweep(configs, out_path, procs=None, resume=True):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = _done_ids(out_path) if resume else set()
    todo = [c for c in (_full(c) for c in configs) if run_id(c) not in done]
    print(f"{len(todo)} runs to do ({len(done)} already on disk) -> {out_path}",
          flush=True)
    if not todo:
        return
    procs = procs or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    with open(out_path, "a") as f, Pool(procs) as pool:
        for i, kpi in enumerate(pool.imap_unordered(_worker, todo), 1):
            f.write(json.dumps(kpi) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {kpi['mode']} n={kpi['n']} T={kpi['temp']:.2f} "
                  f"-> distinct={kpi['n_distinct']}/{kpi['n_solutions_total']} "
                  f"flat={kpi['flat_frac']:.3f} Vmin={kpi['V_min']:.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


def load(path):
    import numpy as np
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    spec, out = sys.argv[1], sys.argv[2]
    mod_path, name = spec.split(":")
    ns = {}
    exec(open(mod_path).read(), ns)
    sweep(expand(ns[name]), out, procs=int(sys.argv[3]) if len(sys.argv) > 3 else None)
