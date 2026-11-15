"""
Niveles 1, 2, 3 partiendo SOLO de las toroidales de un tramo, con dos medidas por nivel:

  cota:   mitades distintas de k filas vistas / P_k(p)   (polinomios exactos, medidor.py)
  origen: que pendientes hay en la genealogia de cada solucion nueva (se guarda la mas corta)
          y de que nivel son las dos piezas que se fusionan.

Uso: python nivel2.py 11 13 17 --niveles 3 --tope 400000
"""
import argparse, time
from collections import Counter, defaultdict
from fusion import wave_gates
from medidor import Pk, DESDE
from nivel1 import sg


def fuse_tagged(have, N, gates):
    """have: {sol: (nivel, pendientes)} -> nuevas {sol: (pendientes, (nivel A, nivel B))}"""
    full = (1 << N) - 1
    L = list(have)
    new = {}
    for G in gates:
        Gs = sorted(G); Cs = [i for i in range(N) if i not in G]
        index = defaultdict(list); part = []
        for s in L:
            cm = d1 = d2 = 0
            for i in Cs:
                c = s[i]; cm |= 1 << c; d1 |= 1 << (i + c); d2 |= 1 << (i - c + N)
            index[cm].append((d1, d2, s))
            cg = e1 = e2 = 0
            for i in Gs:
                c = s[i]; cg |= 1 << c; e1 |= 1 << (i + c); e2 |= 1 << (i - c + N)
            part.append((cg, e1, e2, s))
        for cg, e1, e2, a in part:
            for d1, d2, b in index.get(full ^ cg, ()):
                if (e1 & d1) or (e2 & d2) or b is a:
                    continue
                t = tuple(a[i] if i in G else b[i] for i in range(N))
                if t in have:
                    continue
                tag = have[a][1] | have[b][1]
                lv = tuple(sorted((have[a][0], have[b][0])))
                if t not in new or len(tag) < len(new[t][0]):
                    new[t] = (tag, lv)
    return new


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int, nargs='+')
    ap.add_argument('--niveles', type=int, default=2)
    ap.add_argument('--tope', type=int, default=60000, help='no fusionar un nivel con mas piezas que esto')
    a = ap.parse_args()
    for p in a.N:
        t0 = time.time()
        ks = [k for k in (2, 3, 4, 5, 6) if p >= DESDE[k]]
        have = {tuple((s * i + c) % p for i in range(p)): (0, frozenset([sg(s, p)])) for s in range(2, p - 1) for c in range(p)}
        gates = list(wave_gates(p))
        print(f"===== p={p}   cotas: " + "  ".join(f"P_{k}={Pk(p, k)}" for k in ks))
        print(f"  nivel | soluciones | " + " | ".join(f"cota k={k}" for k in ks))
        for lev in range(a.niveles + 1):
            m = {k: len({s[:k] for s in have}) / Pk(p, k) for k in ks}
            print(f"  {lev:5d} | {len(have):10d} | " + " | ".join(f"{100*m[k]:7.1f}%" for k in ks) + f"   ({time.time()-t0:.0f}s)", flush=True)
            if lev == a.niveles or len(have) > a.tope:
                break
            new = fuse_tagged(have, p, gates)
            if not new:
                break
            nsl = Counter(len(tag) for tag, lv in new.values())
            par = Counter(lv for tag, lv in new.values())
            opp = sum(all(-x in tag for x in tag) for tag, lv in new.values())
            print(f"        nivel {lev+1}: +{len(new)} | n de pendientes distintas en la genealogia: {dict(sorted(nsl.items()))}"
                  f" | solo pares opuestos: {opp} | niveles de las dos piezas: {dict(sorted(par.items()))}")
            if lev == 1:
                pairs = Counter(tuple(sorted({abs(x) for x in tag})) for tag, lv in new.values())
                print("        nivel 2 por pendientes (en valor absoluto): " + "  ".join(f"{k}:{v}" for k, v in sorted(pairs.items(), key=lambda kv: -kv[1])[:14]))
            for t, (tag, lv) in new.items():
                have[t] = (lev + 1, tag)
