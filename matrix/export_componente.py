"""Exporta la componente de 46 (N=8, involución): "a b rango_T". Uso: python3 export_componente.py > g46.txt"""
from collections import Counter, defaultdict
import numpy as np
from experiments import G, involution, UF
g=G(8); mask=involution(g)&g.usable
src,dst,key=g.src[mask],g.dst[mask],g.Tkey[mask]
uf=UF(g.m)
for a,b in zip(src,dst): uf.union(int(a),int(b))
lab=uf.labels(); comp=[int(a) for a in np.where(lab==lab[0])[0]]
pos={a:i for i,a in enumerate(comp)}
nbr=defaultdict(dict)
for a,b,k in zip(src,dst,key): nbr[int(a)][int(k)]=int(b)
cnt=Counter(k for a in comp for k in nbr[a]); keys=sorted(cnt,key=lambda k:(-cnt[k],k)); rank={k:i for i,k in enumerate(keys)}
print(len(comp))
for a in comp:
    for k,b in nbr[a].items(): print(pos[a],pos[b],rank[k])
