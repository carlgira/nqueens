"""Clase 4 y 5 de p=23 (desde data/clases_toro0_p23.txt): orbitas del grupo completo y tipo de cobertura."""
import sys
from collections import Counter, defaultdict
from orbitas import canon_tr, orbits, min_covers
from tipos import tipo, nombre
p = 23; cl = int(sys.argv[1]) if len(sys.argv) > 1 else 4
reps = {}
for ln in open("data/clases_toro0_p23.txt"):
    v = ln.split()
    if v[0] == str(cl): reps[canon_tr([int(x) for x in v[1:]], p)] = cl
orbs, _ = orbits(p, reps)
print(f"p=23 clase {cl}: {len(reps)} orbitas de traslacion, {len(orbs)} orbitas del grupo completo (orden 176)")
res = Counter(); nres = Counter()
for o in orbs:
    T = list(o['rep']); k, covs = min_covers(T, p)
    ts = sorted({t for t in (tipo(cv, T, p) for cv in covs) if t})
    if not ts:
        sl = Counter(a for a, _, _ in covs[0]); ts = ['otra: ' + '+'.join(map(str, sorted(sl.values(), reverse=True))) + ' pendientes [' + ','.join(sorted(nombre(a, p) for a in sl)) + ']']
    key = ' / '.join(ts) + ('' if len(covs) == 1 else f'  ({len(covs)} coberturas)')
    res[key] += o['tam']; nres[key] += 1
for k, v in sorted(res.items(), key=lambda x: -x[1]): print(f"  {v:5d} orbitas de traslacion en {nres[k]:2d} orbitas completas: {k}")
