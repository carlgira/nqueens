"""Do chaotic maps produce n-queens solutions more often than chance?

Search formula space instead of parameter space.  Each family is a recipe that
turns parameters into a permutation; we generate a lot of them and measure the
fraction that are solutions, against the only honest null: the density of
solutions among all permutations, Q(N)/N!.

The measure that matters is the ENRICHMENT, hit rate / density:
  ~1   the family knows nothing, it is random search wearing a costume
  >>1  the family's structure is aligned with the constraints

Chaotic orbits are turned into permutations by rank order (argsort), so every
sample is a permutation and the null applies exactly.  Zero-entropy families
(rotations, linear/quadratic mod N) are included as the comparison, and a
uniform random permutation is included to verify the measurement returns 1.
"""

import math

import numpy as np

from .core import enumerate_solutions

K = None  # row indices, set per N


def is_solution(C):
    """C: (M, N) array of permutations -> (M,) bool.  Columns are given."""
    n = C.shape[1]
    k = np.arange(n)
    out = np.ones(C.shape[0], bool)
    for d in (C + k, C - k):
        s = np.sort(d, axis=1)
        out &= (np.diff(s, axis=1) != 0).all(axis=1)
    return out


# ---------------------------------------------------------------- chaotic maps
def _orbit(step, x0, n):
    """Iterate `step` n times from each x0, return the (M, N) trajectory."""
    x = x0.copy()
    cols = np.empty((x0.size, n))
    for i in range(n):
        cols[:, i] = x
        x = step(x)
    return cols


def fam_logistic(n, m, rng):
    r = rng.uniform(3.9, 4.0, m)
    return np.argsort(_orbit(lambda x: r * x * (1 - x), rng.uniform(.05, .95, m), n), axis=1)


def fam_tent(n, m, rng):
    r = rng.uniform(1.9, 2.0, m)
    return np.argsort(_orbit(lambda x: r * np.minimum(x, 1 - x),
                             rng.uniform(.05, .95, m), n), axis=1)


def fam_bernoulli(n, m, rng):
    return np.argsort(_orbit(lambda x: (2 * x) % 1.0, rng.uniform(0, 1, m), n), axis=1)


def fam_sine(n, m, rng):
    r = rng.uniform(0.95, 1.0, m)
    return np.argsort(_orbit(lambda x: r * np.sin(np.pi * x),
                             rng.uniform(.05, .95, m), n), axis=1)


def fam_cat(n, m, rng):
    """Arnold's cat map, read the x coordinate."""
    x, y = rng.uniform(0, 1, m), rng.uniform(0, 1, m)
    cols = np.empty((m, n))
    for i in range(n):
        cols[:, i] = x
        x, y = (x + y) % 1.0, (x + 2 * y) % 1.0
    return np.argsort(cols, axis=1)


def fam_standard(n, m, rng):
    """Chirikov standard map, the textbook chaotic Hamiltonian."""
    kk = rng.uniform(1.0, 6.0, m)
    th, p = rng.uniform(0, 1, m), rng.uniform(0, 1, m)
    cols = np.empty((m, n))
    for i in range(n):
        cols[:, i] = th
        p = (p + kk / (2 * np.pi) * np.sin(2 * np.pi * th)) % 1.0
        th = (th + p) % 1.0
    return np.argsort(cols, axis=1)


def fam_henon(n, m, rng):
    a, b = 1.4, 0.3
    x, y = rng.uniform(-.5, .5, m), rng.uniform(-.5, .5, m)
    cols = np.empty((m, n))
    for i in range(n):
        cols[:, i] = x
        x, y = 1 - a * x * x + y, b * x
    return np.argsort(cols, axis=1)


# --------------------------------------------------- zero-entropy comparisons
def fam_rotation(n, m, rng):
    """Irrational rotation / Sturmian: rank order of frac(alpha*k + beta)."""
    al = rng.uniform(0, 1, m)[:, None]
    be = rng.uniform(0, 1, m)[:, None]
    return np.argsort((al * np.arange(n) + be) % 1.0, axis=1)


def fam_linear(n, m, rng):
    """(a*k + b) mod N -- the mod-N construction.  Only a permutation if gcd(a,N)=1."""
    a = rng.integers(1, n, m)[:, None]
    b = rng.integers(0, n, m)[:, None]
    return (a * np.arange(n) + b) % n


def fam_quadratic(n, m, rng):
    a = rng.integers(0, n, m)[:, None]
    b = rng.integers(1, n, m)[:, None]
    c = rng.integers(0, n, m)[:, None]
    k = np.arange(n)
    return (a * k * k + b * k + c) % n


def fam_power(n, m, rng):
    a = rng.integers(1, n, m)[:, None]
    e = rng.integers(1, max(n - 1, 2), m)[:, None]
    k = np.arange(1, n + 1)
    return (a * np.power(k, e, dtype=object).astype(np.int64) % n)


def fam_random(n, m, rng):
    """The null itself.  Enrichment must come out at 1.0 or the metric is wrong."""
    return np.argsort(rng.random((m, n)), axis=1)


FAMILIES = [
    ("logistic", fam_logistic, "caotica"),
    ("tent", fam_tent, "caotica"),
    ("bernoulli", fam_bernoulli, "caotica"),
    ("sine", fam_sine, "caotica"),
    ("cat map", fam_cat, "caotica"),
    ("standard map", fam_standard, "caotica"),
    ("henon", fam_henon, "caotica"),
    ("rotacion/Sturmian", fam_rotation, "entropia 0"),
    ("lineal mod N", fam_linear, "entropia 0"),
    ("cuadratica mod N", fam_quadratic, "entropia 0"),
    ("potencia mod N", fam_power, "entropia 0"),
    ("permut. aleatoria", fam_random, "control"),
]


def is_perm(C):
    n = C.shape[1]
    return np.array([len(set(r.tolist())) == n for r in C])


def evaluate(n, m=200_000, seed=0):
    dens = len(enumerate_solutions(n, False)) / math.factorial(n)
    rng = np.random.default_rng(seed)
    out = []
    for name, fn, kind in FAMILIES:
        C = np.asarray(fn(n, m, rng), dtype=np.int64)
        perm = is_perm(C)
        P = C[perm]
        if len(P) == 0:
            out.append((name, kind, 0, 0, 0, float("nan")))
            continue
        sol = is_solution(P)
        hit = sol.mean()
        distinct = len({tuple(r) for r in P[sol]})
        out.append((name, kind, perm.mean(), hit, distinct, hit / dens))
    return dens, out
