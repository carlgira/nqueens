"""Exact enumeration of the irrational-rotation (Sturmian) family.

c_k = rank of frac(alpha*k + beta) among k = 0..N-1.

The repertoire is finite and small, so it can be enumerated completely instead
of sampled.  Two points frac(alpha*k) and frac(alpha*j) swap order exactly when
alpha crosses m/|k-j|, so the permutation is constant on the open intervals
between Farey fractions of order N-1.  For a representative alpha inside each
interval, beta then has exactly N regimes, split at beta = 1 - frac(alpha*k).
Taking a midpoint of every (alpha, beta) cell enumerates the family exactly.

    python -m wave_queens.rotation
"""

from fractions import Fraction
from math import gcd

import numpy as np

from .core import canonical, enumerate_solutions


def farey(n):
    """Every p/q in [0,1] in lowest terms with q <= n, ascending."""
    return sorted({Fraction(p, q) for q in range(1, n + 1) for p in range(q + 1)})


def repertoire(n):
    """Every permutation the family can emit at this N, with its (alpha, beta) cell."""
    fs = farey(max(n - 1, 1))
    alphas = [(float(a) + float(b)) / 2 for a, b in zip(fs, fs[1:])]
    out = {}
    for a in alphas:
        v = (a * np.arange(n)) % 1.0
        cuts = np.sort((1.0 - v) % 1.0)
        betas = [(x + y) / 2 for x, y in zip(cuts, np.r_[cuts[1:], cuts[0] + 1.0])]
        for b in betas:
            x = (v + b) % 1.0
            c = tuple(int(i) for i in np.argsort(np.argsort(x)))
            out.setdefault(c, (a, b % 1.0))
    return out


# canonicalising every solution costs more than the whole enumeration once Q(N)
# passes ~10^5, so the family-coverage column is only filled where it is cheap.
CHEAP_CLASSES = 12


def solves(c):
    """Test one permutation directly.  Building the full solution set instead
    was the bottleneck and made N>14 impossible; Q(20) is 1.4e9."""
    n = len(c)
    return (len({c[i] + i for i in range(n)}) == n
            and len({c[i] - i for i in range(n)}) == n)


def analyse(n, want_classes=True):
    rep = repertoire(n)
    hits = {c: ab for c, ab in rep.items() if solves(c)}
    classes = {canonical(np.array(c), n, False) for c in hits}
    all_classes = ({canonical(np.array(s), n, False)
                    for s in enumerate_solutions(n, False)}
                   if want_classes and n <= CHEAP_CLASSES else None)
    return dict(n=n, repertoire=len(rep), hits=len(hits),
                frac=len(hits) / len(rep) if rep else 0.0,
                classes=len(classes),
                all_classes=len(all_classes) if all_classes is not None else None,
                cells=hits)


def alpha_structure(res, top=6):
    """Do the winning alphas sit near simple rationals?"""
    n = res["n"]
    out = []
    for c, (a, b) in sorted(res["cells"].items(), key=lambda kv: kv[1][0]):
        best = min(((abs(a - p / q), p, q) for q in range(2, n + 2)
                    for p in range(1, q) if gcd(p, q) == 1), key=lambda t: t[0])
        out.append((a, b, c, f"{best[1]}/{best[2]}", best[0]))
    return out[:top]


if __name__ == "__main__":
    print(f"{'N':>3}{'repertorio':>12}{'soluciones':>12}{'frac':>9}"
          f"{'de Q(N)':>10}{'familias':>12}{'vs densidad':>13}")
    import math
    keep = {}
    for n in range(7, 15):
        r = analyse(n)
        keep[n] = r
        dens = r["total"] / math.factorial(n)
        fam = f"{r['classes']}/{r['all_classes']}" if r['all_classes'] else f"{r['classes']}/?"
        print(f"{n:>3}{r['repertoire']:>12,}{r['hits']:>12}{r['frac']:>9.4f}"
              f"{r['hits']/r['total']*100:>9.2f}%{fam:>12}{r['frac']/dens:>12,.0f}x")

    print("\nalphas ganadores (N=10, donde la construccion mod-N no existe):")
    for a, b, c, near, err in alpha_structure(keep[10], top=12):
        print(f"   alpha={a:.5f}  beta={b:.5f}  ~ {near:>6} (err {err:.4f})   {c}")
