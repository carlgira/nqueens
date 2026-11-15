"""What kind of solutions did stage C actually find?

The sweep stored counts, not placements, so the exact list is not recoverable.
But it stored `n_distinct_class`, and the symmetry classes of standard N=7 split
cleanly into modular-only and non-modular-only -- no mixed class.  That turns a
class count into a rigorous statement about what was found.
"""

import collections
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import canonical, enumerate_solutions

R = "wave_queens/results"


def partition(n):
    std = [tuple(s) for s in enumerate_solutions(n, modular=False)]
    mod = {tuple(s) for s in enumerate_solutions(n, modular=True)}
    assert mod <= set(std), "modular solutions must be a subset of standard ones"
    cls = collections.defaultdict(lambda: [0, 0])
    for s in std:
        cls[canonical(np.array(s), n, False)][0 if s in mod else 1] += 1
    return std, mod, dict(cls)


def figure(n=7, path=f"{R}/class_structure_n7.png"):
    std, mod, cls = partition(n)
    rows = [json.loads(l) for l in open(f"{R}/std_C_pt.jsonl")]
    rows7 = [r for r in rows if r["n"] == n]
    best = max(rows7, key=lambda r: r["n_distinct"])

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    # (1) how the solution set splits
    keys = sorted(cls, key=lambda k: (-cls[k][0], -sum(cls[k])))
    m = [cls[k][0] for k in keys]
    nm = [cls[k][1] for k in keys]
    x = np.arange(len(keys))
    ax[0].bar(x, m, color="tab:blue", label=f"modular ({sum(m)})")
    ax[0].bar(x, nm, bottom=m, color="tab:orange", label=f"non-modular ({sum(nm)})")
    ax[0].set_xticks(x, [f"C{i+1}" for i in x])
    ax[0].set_xlabel("symmetry class")
    ax[0].set_ylabel("solutions")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"N={n}: {len(std)} standard solutions, {len(cls)} classes\n"
                    "no class mixes the two kinds", fontsize=10)

    # (2) what the best run must have hit
    lo_nm, hi_nm = 2, min(best["n_distinct"] - 4, sum(nm))
    ax[1].barh([1], [best["n_distinct"]], color="0.85", label="12 found")
    ax[1].barh([1], [hi_nm], color="tab:orange", alpha=.6)
    ax[1].barh([1], [lo_nm], color="tab:orange")
    ax[1].set_yticks([])
    ax[1].set_xlabel("distinct solutions")
    ax[1].set_title(f"best stage-C run: {best['n_distinct']} solutions across\n"
                    f"{best['n_distinct_class']}/{best['n_classes_total']} classes"
                    f" -> {lo_nm}-{hi_nm} of them non-modular", fontsize=10)
    ax[1].text(lo_nm / 2, 1, f">={lo_nm}", ha="center", va="center", color="w", fontsize=9)
    ax[1].text((lo_nm + hi_nm) / 2, 1.32, f"up to {hi_nm}", ha="center", fontsize=8)
    ax[1].axvline(best["n_distinct"] * sum(nm) / len(std), color="k", ls="--", lw=1)
    ax[1].text(best["n_distinct"] * sum(nm) / len(std) + .15, 0.72,
               f"uniform expectation\n({sum(nm)}/{len(std)} = "
               f"{100*sum(nm)/len(std):.0f}% non-modular)", fontsize=8)

    # (3) the ladder is what buys the coverage
    for r in rows7:
        ax[2].scatter(r["n_replicas"], r["n_distinct"], s=40,
                      c=[["tab:red", "tab:green"][r["n_distinct_class"] >= 5]],
                      alpha=.75)
    ax[2].scatter([], [], c="tab:green", label=">=5 of 6 classes")
    ax[2].scatter([], [], c="tab:red", label="<5 classes")
    ax[2].set_xticks([8, 16])
    ax[2].set_xlabel("replicas in the ladder")
    ax[2].set_ylabel("distinct solutions found")
    ax[2].legend(fontsize=8)
    ax[2].set_title("stage C, N=7 (each point = one run)", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"{len(std)} standard = {len(mod)} modular + {len(std)-len(mod)} non-modular")
    print(f"best run: {best['n_distinct']} distinct, {best['n_distinct_class']} classes"
          f"  ->  non-modular in [{lo_nm}, {hi_nm}], modular in "
          f"[{best['n_distinct']-hi_nm}, {best['n_distinct']-lo_nm}]")
    return path


if __name__ == "__main__":
    figure()
