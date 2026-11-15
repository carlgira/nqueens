"""Regimenes del rio: repetir una solucion o encadenar soluciones distintas.

  python experiments.py chain   N        -> cadena de soluciones de una orbita
  python experiments.py regimes N        -> barrido (beta, lam): distintas, repetidas
  python experiments.py basins  N        -> que subconjunto visita cada condicion inicial
"""
import sys
import itertools
import numpy as np
from core import TorusFlow, is_solution


def chain(laps):
    """Solucion por vuelta (None si no lo es) y la secuencia de soluciones sin duplicados consecutivos."""
    seq = []
    for l in laps:
        if is_solution(list(l)):
            if not seq or seq[-1] != l:
                seq.append(l)
    return seq


def summary(laps):
    hits = [l for l in laps if is_solution(list(l))]
    seq = chain(laps)
    first_repeat = None
    seen = set()
    for k, s in enumerate(seq):
        if s in seen:
            first_repeat = k
            break
        seen.add(s)
    return dict(hit_rate=len(hits) / len(laps), distinct=len(set(hits)),
                links=len(seq), first_repeat=first_repeat)


def cmd_chain(n, laps=300, **kw):
    tf = TorusFlow(n=n, **kw)
    L = tf.run(laps)
    for k, l in enumerate(L):
        ok = is_solution(list(l))
        print(f"{k:4d} {''.join('0123456789ABCDEFGHIJ'[c] for c in l)} {'*' if ok else ''}")
    print(summary(L))


def cmd_regimes(n, laps=800, lams=(0.005, 0.02, 0.05, 0.15)):
    """beta se da via beta*M*, con M* = 1/(1-e^{-lam N}) la carga de saturacion de una celda."""
    print("lam     beta*M*  aciertos  distintas  eslabones  primera_repeticion")
    for lam in lams:
        for f in (0.5, 0.8, 1.0, 1.2, 1.5, 2.0):
            beta = f * (1 - np.exp(-lam * n))
            s = summary(TorusFlow(n=n, beta=beta, lam=lam, x0=0.3).run(laps))
            print(f"{lam:6.3f}  {f:5.2f}    {s['hit_rate']:6.2f}   {s['distinct']:6d}   "
                  f"{s['links']:6d}     {s['first_repeat']}", flush=True)


def cmd_basins(n, laps=300, beta=0.3, lam=0.02, k=12):
    sets = {}
    for x0 in np.linspace(0, n, k, endpoint=False):
        for alpha in (2.0, 2.5, 3.0):
            seq = chain(TorusFlow(n=n, beta=beta, lam=lam, x0=x0, alpha=alpha).run(laps))
            sets[(round(x0, 2), alpha)] = seq
            print(f"x0={x0:5.2f} alpha={alpha}: {len(seq):3d} eslabones, {len(set(seq)):3d} distintas")
    allsols = set().union(*map(set, sets.values()))
    print("union de todas las orbitas:", len(allsols), "soluciones distintas")


if __name__ == "__main__":
    cmd, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if cmd == "regimes" and len(sys.argv) > 3:
        cmd_regimes(n, lams=tuple(float(a) for a in sys.argv[3:]))
    else:
        {"chain": cmd_chain, "regimes": cmd_regimes, "basins": cmd_basins}[cmd](n)
