"""Para cada orbita del grupo completo: todas las coberturas minimas del representante."""
import sys, json
from collections import Counter
from orbitas import min_covers, inv

def rows(m, p): return [r for r in range(p) if m >> r & 1]

for p in [int(x) for x in sys.argv[1:]]:
    for o in json.load(open(f"data/orbitas_p{p}.json")):
        if o['clase'] < 3: continue
        T = o['rep']; k, covs = min_covers(T, p)
        pats = Counter(tuple(sorted(a for a, c, m in cv)) for cv in covs)
        print(f"p={p} clase {k} orbita tam {o['tam']} (estab {o['stab']})  T={T}")
        print(f"   {len(covs)} coberturas minimas; pendientes: {dict(pats)}")
        if '-v' in sys.argv or k == 3:
            for cv in covs:
                print("     ", "  ".join(f"[a={a} c={c} filas={rows(m,p)}]" for a, c, m in cv))
