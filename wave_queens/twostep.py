"""Generalised two-step family: the rotation family with the step gap freed.

The rotation family emits step sequences whose two values are ADJACENT (gap 1) --
that is what the three-distance theorem gives.  It therefore cannot reach
N = 2 mod 6, whose simplest solutions have two steps separated by 3
(N=8: {2,5},{3,6};  N=14: {7,10}).

So keep the Sturmian *pattern* -- which of the two steps to take at each k comes
from a mechanical word, exactly as in the rotation family -- but let the two step
VALUES be any pair (p, q) in Z_N.  This strictly contains the rotation family
(q = p+1) and is still zero entropy: a rotation drives the choice.

    c_0 = 0,   c_{k+1} = c_k + (p if w_k = 0 else q)  mod N
    w   = mechanical word of the rotation (alpha, beta)

c_0 is fixed at 0 because translating every column preserves both diagonal
conditions, so it adds nothing but copies.

    python -m wave_queens.twostep
"""

import numpy as np

from .rotation import farey


def mechanical_words(n):
    """Every binary word of length n-1 a rotation can cut, with its (alpha, beta)."""
    fs = farey(max(n - 1, 1))
    out = {}
    for a, b in zip(fs, fs[1:]):
        al = (float(a) + float(b)) / 2
        v = (al * np.arange(n)) % 1.0
        cuts = np.sort((1.0 - v) % 1.0)
        for x, y in zip(cuts, np.r_[cuts[1:], cuts[0] + 1.0]):
            be = (x + y) / 2
            f = np.floor(al * np.arange(n) + be)
            w = tuple(int(t) for t in np.diff(f))
            out.setdefault(w, (al, be % 1.0))
    return out


def solves(c, n):
    return (len(set(c)) == n
            and len({c[i] + i for i in range(n)}) == n
            and len({c[i] - i for i in range(n)}) == n)


def search(n):
    """All (p, q, word) whose sequence is a solution."""
    words = mechanical_words(n)
    hits = {}
    for w, (al, be) in words.items():
        w = np.array(w)
        for p in range(n):
            for q in range(n):
                s = np.where(w == 0, p, q)
                c = tuple(int(x) for x in np.r_[0, np.cumsum(s)] % n)
                if len(c) == n and solves(c, n):
                    hits.setdefault(c, (p, q, al, be))
    return hits, len(words)


if __name__ == "__main__":
    print(f"{'N':>4}{'N mod 6':>9}{'palabras':>10}{'aciertos':>10}"
          f"{'gaps |q-p| que funcionan':>28}")
    for n in (8, 9, 10, 11, 12, 14, 20, 26):
        hits, nw = search(n)
        gaps = sorted({min(abs(v[1] - v[0]), n - abs(v[1] - v[0]))
                       for v in hits.values()})
        print(f"{n:>4}{n % 6:>9}{nw:>10}{len(hits):>10}{str(gaps):>28}")

    print("\nN=8 (clase 2, donde rotacion y modular dan CERO):")
    hits, _ = search(8)
    for c, (p, q, al, be) in sorted(hits.items())[:8]:
        st = sorted({(c[i + 1] - c[i]) % 8 for i in range(7)})
        print(f"   {c}   p={p} q={q}  pasos {st}  alpha={al:.4f}")
