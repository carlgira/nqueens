"""Banco de criterios sobre el grafo completo graph_N.npz.

Cada criterio es una función  criterio(G) -> máscara booleana sobre las aristas (o una lista de aristas).
Para cada criterio se mide el subgrafo resultante (componentes, tamaños, pureza respecto a órbitas D4).

Uso: python experiments.py N [N ...]
Para añadir un criterio: define una función y regístrala en CRITERIA.
"""
import sys
from collections import Counter
import numpy as np
from build_graph import is_toroidal


# ---------- carga ----------
class G:
    def __init__(self, n):
        z = np.load(f"graph_{n}.npz")
        self.n = int(z["n"]); self.sols = z["sols"]; self.sym = z["sym"]
        self.src = z["src"]; self.dst = z["dst"]
        self.hamming = z["hamming"]; self.cayley = z["cayley"]
        self.ctype = z["ctype"]; self.ncyc = z["ncyc"]; self.fixed = z["fixed"]
        self.ctype_names = z["ctype_names"]
        self.m = len(self.sols); self.E = len(self.src)
        # T exacta de cada arista (como permutación), clave = bytes
        T = np.empty((self.E, self.n), dtype=np.int8)
        rows = np.arange(self.E)[:, None]
        T[rows, self.sols[self.src]] = self.sols[self.dst]
        self.T = T
        # clave entera exacta (base N): evita los problemas de numpy con bytes que acaban en \x00
        self.Tkey = (T.astype(np.int64) * (self.n ** np.arange(self.n, dtype=np.int64))).sum(axis=1)
        self.orbit = self._orbits()
        # aristas útiles: se descartan las que unen una solución con su propia imagen por simetría D4
        self.nosym = self.orbit[self.src] != self.orbit[self.dst]
        # reglas del grafo: (1) no unir simetrías D4  (2) no unir dos toroidales entre sí
        self.tor = is_toroidal(self.sols)
        self.usable = self.nosym & ~(self.tor[self.src] & self.tor[self.dst])

    def _orbits(self):
        uf = UF(self.m)
        for a in range(self.m):
            for b in self.sym[a]:
                uf.union(a, int(b))
        return uf.labels()


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb

    def labels(self):
        roots = [self.find(i) for i in range(len(self.p))]
        _, lab = np.unique(roots, return_inverse=True)
        return lab


# ---------- métricas ----------
def measure(g, src, dst):
    uf = UF(g.m)
    for a, b in zip(src, dst):
        uf.union(int(a), int(b))
    lab = uf.labels()
    sizes = np.bincount(lab)
    comp_orbits = {}
    for node, c in enumerate(lab):
        comp_orbits.setdefault(c, set()).add(int(g.orbit[node]))
    big = [len(s) for c, s in comp_orbits.items() if sizes[c] > 1]
    orb_mean = float(np.mean(big)) if big else 0.0     # órbitas D4 distintas por componente no trivial
    return dict(edges=len(src), comps=len(sizes), largest=int(sizes.max()),
                singletons=int((sizes == 1).sum()), orb_mean=orb_mean,
                sizes=sorted(sizes.tolist(), reverse=True))


def report(name, g, src, dst, extra=""):
    r = measure(g, src, dst)
    s = r["sizes"]
    shown = s if len(s) <= 12 else s[:12] + ["..."]
    print(f"  {name:<34} aristas={r['edges']:>7}  comps={r['comps']:>4}  "
          f"mayor={r['largest']:>4}  aislados={r['singletons']:>4}  "
          f"órbitas/comp={r['orb_mean']:>5.1f}  tamaños={shown} {extra}")
    return r


# ---------- criterios: devuelven máscara sobre aristas ----------
def hamming_le(k):
    return lambda g: g.hamming <= k

def cayley_le(k):
    return lambda g: g.cayley <= k

def derangement(g):            # T sin puntos fijos
    return g.fixed == 0

def involution(g):             # T^2 = id (ciclos de longitud 1 y 2)
    ok = np.array([max(eval(s)) <= 2 for s in g.ctype_names])
    return ok[g.ctype]

def single_cycle(g):           # T es un ciclo de longitud N
    return g.ncyc == 1

def symmetry_edges(g):         # a -> b si b es imagen de a por D4
    m = np.zeros(g.E, dtype=bool)
    sym_sets = [set(map(int, r)) for r in g.sym]
    for e, (a, b) in enumerate(zip(g.src, g.dst)):
        m[e] = int(b) in sym_sets[int(a)]
    return m

CRITERIA = {
    "hamming<=4": hamming_le(4),
    "hamming<=5": hamming_le(5),
    "cayley<=2": cayley_le(2),
    "cayley<=3": cayley_le(3),
    "T derangement": derangement,
    "T involución": involution,
    "T ciclo único de longitud N": single_cycle,
}


# ---------- "misma T": la arista tiene exactamente la misma matriz de transformación ----------
def same_T_experiment(g, top=5):
    """Para cada T concreta que aparece en varias aristas, el subgrafo con SOLO esa T es un grafo
    funcional (cada nodo sale a lo sumo una vez): caminos y ciclos. Es el criterio 'generable':
    una sola matriz T que, aplicada repetidamente, recorre un conjunto de soluciones."""
    cnt = Counter(g.Tkey[g.usable].tolist())
    rep = [(k, c) for k, c in cnt.most_common() if c > 1]
    print(f"  T exactas distintas={len(cnt)}, usadas en >1 arista={len(rep)}, "
          f"máx repeticiones={cnt.most_common(1)[0][1]}")
    for key, c in cnt.most_common(top):
        idx = np.where((g.Tkey == key) & g.usable)[0]
        t = [(key // g.n ** i) % g.n for i in range(g.n)]
        report(f"misma T={t}", g, g.src[idx], g.dst[idx])
    # cuántas aristas quedan cubiertas si juntamos todas las T repetidas
    allidx = np.where(np.isin(g.Tkey, [k for k, _ in rep]))[0]
    return len(cnt), len(rep)


# ---------- algoritmos de caminos (no máscaras) ----------
def lexicographic_path(g):
    """Camino secuencial: soluciones en orden lexicográfico, i -> i+1. Mide el coste de cada salto."""
    order = np.lexsort(g.sols.T[::-1])
    ham = [(g.sols[order[i]] != g.sols[order[i + 1]]).sum() for i in range(g.m - 1)]
    return order, np.array(ham)

def greedy_path(g, metric="hamming", start=0):
    """Vecino más cercano: desde el nodo actual, ir al no visitado con menor distancia."""
    D = np.full((g.m, g.m), 99, dtype=np.int16)
    ns = g.usable
    D[g.src[ns], g.dst[ns]] = getattr(g, metric)[ns]
    visited = np.zeros(g.m, bool); cur = start; path = [cur]; visited[cur] = True; cost = []
    for _ in range(g.m - 1):
        row = D[cur].copy(); row[visited] = 999
        nxt = int(row.argmin()); cost.append(int(row[nxt]))
        visited[nxt] = True; path.append(nxt); cur = nxt
    return path, np.array(cost)


def run(n):
    g = G(n)
    n_orb = len(set(g.orbit.tolist()))
    print(f"\n=== N={n}: {g.m} soluciones ({int(g.tor.sum())} toroidales), {g.E} aristas "
          f"({int(g.usable.sum())} usables), {n_orb} órbitas D4 ===")
    print("Criterios por máscara:")
    for name, fn in CRITERIA.items():
        mask = fn(g) & g.usable
        report(name, g, g.src[mask], g.dst[mask])
    print("Criterio 'misma T':")
    same_T_experiment(g)
    print("Caminos:")
    order, ham = lexicographic_path(g)
    print(f"  orden lexicográfico: salto Hamming medio={ham.mean():.2f}, min={ham.min()}, max={ham.max()}")
    for metric in ("hamming", "cayley"):
        p, c = greedy_path(g, metric)
        dead = int((c >= 99).sum())          # saltos forzados: solo quedaban imágenes simétricas
        ok = c[c < 99]
        print(f"  vecino más cercano ({metric}): coste medio={ok.mean():.2f} (sin {dead} callejones), "
              f"saltos con coste<=3: {(ok <= 3).sum()}/{len(c)}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        run(int(a))
