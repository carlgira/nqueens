# Comunidades (Louvain, implementación propia) en la red de co-aparición de soluciones
# en bucles con >= 4 soluciones (datos/apuntado_N8_<tag>_MM4.txt). Peso = nº de firmas compartidas.
import collections, random, sys
def load(tag):
    E=collections.Counter(); sig=collections.defaultdict(list)
    for l in open(f'datos/apuntado_N8_{tag}_MM4.txt'):
        if l.startswith('#'): continue
        r=l.rstrip('\n').split(' | '); g=sorted(set(map(int,r[6].split())))
        for i in range(len(g)):
            for j in range(i+1,len(g)): E[(g[i],g[j])]+=1
        for x in g: sig[x].append((r[1],r[2].strip()))
    return E,sig
def modularity(adj,part,m2):
    k={u:sum(adj[u].values()) for u in adj}; Q=0
    tot=collections.Counter(); inn=collections.Counter()
    for u in adj:
        tot[part[u]]+=k[u]
        for v,w in adj[u].items():
            if part[u]==part[v]: inn[part[u]]+=w
    return sum(inn[c]/m2-(tot[c]/m2)**2 for c in tot)
def louvain(adj,seed):
    rnd=random.Random(seed); nodes=list(adj); m2=sum(sum(a.values()) for a in adj.values())
    # una sola fase de movimientos locales + agregación repetida
    part={u:u for u in nodes}; level=[(u,) for u in nodes]
    groups={u:{u} for u in nodes}
    A={u:dict(adj[u]) for u in adj}
    while True:
        k={u:sum(A[u].values())+A[u].get(u,0) for u in A}
        com={u:u for u in A}; tot=collections.Counter({u:k[u] for u in A})
        moved=True; anym=False
        while moved:
            moved=False; order=list(A); rnd.shuffle(order)
            for u in order:
                cu=com[u]; tot[cu]-=k[u]
                wc=collections.Counter()
                for v,w in A[u].items():
                    if v!=u: wc[com[v]]+=w
                best=cu; bg=wc.get(cu,0)-tot[cu]*k[u]/m2
                for c,w in wc.items():
                    g=w-tot[c]*k[u]/m2
                    if g>bg+1e-12: bg=g; best=c
                com[u]=best; tot[best]+=k[u]
                if best!=cu: moved=True; anym=True
        if not anym: break
        # agregar
        newg=collections.defaultdict(set)
        for u in A: newg[com[u]]|=groups[u]
        NA=collections.defaultdict(collections.Counter)
        for u in A:
            for v,w in A[u].items(): NA[com[u]][com[v]]+=w
        A={c:dict(NA[c]) for c in NA}; groups=dict(newg)
    part={}
    for c,s in groups.items():
        for x in s: part[x]=c
    return part
def run(tag,minw=1):
    E,sig=load(tag)
    adj=collections.defaultdict(dict)
    for (a,b),w in E.items():
        if w>=minw: adj[a][b]=w; adj[b][a]=w
    m2=sum(sum(a.values()) for a in adj.values())
    best=None
    for s in range(30):
        p=louvain(adj,s); q=modularity(adj,p,m2)
        if best is None or q>best[0]: best=(q,p)
    q,p=best; com=collections.defaultdict(list)
    for x,c in p.items(): com[c].append(x)
    sizes=sorted((len(v) for v in com.values()),reverse=True)
    return q,sizes,[sorted(v) for v in sorted(com.values(),key=len,reverse=True)],sig
if __name__=='__main__':
    for tag in ['sol','sim1','sim2']:
        q,sizes,coms,_=run(tag)
        print(f'{tag}: modularidad {q:.3f}  comunidades {len(sizes)}  tamanos {sizes}')
