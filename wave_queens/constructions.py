"""Three closed forms, one per residue class of N mod 6, covering every N.

Found by working backwards from the formula-space search: the rotation family's
hits split by N mod 6, and inspecting the winning step words gave the shapes
below.  Verified for every N from 4 to 600 with no gaps.

This is the classical explicit n-queens construction (Hoffman-Loessi-Moore 1969,
Bernhardsson 1991), which is likewise case-split on N mod 6 -- so the formulas
are not new.  What the search contributed is deriving the case structure from
data rather than assuming it, and showing WHY the split is at 6: the modular
rotation needs gcd(N,6)=1, and the residues it cannot reach are exactly the ones
that need a defect (even N) or a two-block word (N = 3 mod 6).

    python -m wave_queens.constructions 30
"""

import sys
from math import gcd


def solves(c):
    n = len(c)
    return (len(set(c)) == n
            and len({c[i] + i for i in range(n)}) == n
            and len({c[i] - i for i in range(n)}) == n)


def _run(n, c0, steps):
    c = [c0]
    for s in steps:
        c.append((c[-1] + s) % n)
    return tuple(c)


def modular(n):
    """gcd(N,6)=1: constant step.  The classic rotation, zero entropy."""
    for a in range(2, n):
        if gcd(a, n) == 1:
            c = tuple((a * k) % n for k in range(n))
            if solves(c):
                return c
    return None


def even(n):
    """N even: step 2 everywhere, one defect at the midpoint d=(N-2)/2.

    N = 2 mod 6 needs c0=3 and defect 5; N = 0,4 mod 6 need c0=1 and defect 1.
    The defect is what the rotation family cannot express -- its two step values
    are always adjacent, and here they differ by 3 (or by 1 at a single place).
    """
    q, c0 = (5, 3) if n % 6 == 2 else (1, 1)
    s = [2] * (n - 1)
    s[(n - 2) // 2] = q
    return _run(n, c0, s)


def mod6_3(n):
    """N = 3 mod 6: steps {N-3, N-2} driven by the word 00 1^m 00 1^m 00.

    m = (N-7)/2, so the word has length N-1.  This is the only class needing a
    two-block pattern rather than a constant step or a single defect.
    """
    m = (n - 7) // 2
    w = [0, 0] + [1] * m + [0, 0] + [1] * m + [0, 0]
    return _run(n, 1, [n - 3 if b == 0 else n - 2 for b in w])


def construct(n):
    """A solution for any n >= 4, picked by n mod 6."""
    if n < 4:
        return None
    if gcd(n, 6) == 1:
        return modular(n)
    return even(n) if n % 2 == 0 else mod6_3(n)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    c = construct(n)
    which = ("modular" if gcd(n, 6) == 1 else
             ("un defecto" if n % 2 == 0 else "dos bloques"))
    print(f"N={n}  ({n} mod 6 = {n % 6}, familia: {which})")
    print(f"  {c}")
    print(f"  valida: {solves(c)}")
    bad = [m for m in range(4, 401) if not solves(construct(m))]
    print(f"\n  verificacion N=4..400: {'sin fallos' if not bad else bad}")
