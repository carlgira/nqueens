"""¿Qué separa las componentes del criterio 'involución'? Busca un invariante simple.

Para cada N con >=2 componentes grandes, prueba como etiqueta de componente:
  - paridad de una suma enmascarada de q_i  (todas las máscaras 0/1)
  - paridad de la suma enmascarada de [q_i >= N/2]   (mitad izquierda/derecha)
  - lo mismo sobre la permutación inversa (vista por columnas)
y también si cada componente toma la mitad de cada órbita D4.
Uso: python analyze.py 5 7 8
"""
import sys, itertools
import numpy as np
from scan import scan, perm_sign


def best_features(S, lab, n):
    big = [k for k in np.unique(lab) if (lab == k).sum() > 1]
    keep = np.isin(lab, big)
    y = (lab[keep] == big[0]).astype(int) if len(big) == 2 else None
    if y is None:
        return None
    X = S[keep]
    inv = np.argsort(X, axis=1)
    hits = []
    for name, M in (("q_i", X), ("[q_i>=N/2]", (X >= n / 2).astype(int)),
                    ("qinv_j", inv), ("[qinv_j>=N/2]", (inv >= n / 2).astype(int))):
        for mask in itertools.product((0, 1), repeat=n):
            f = (M * np.array(mask)).sum(axis=1) % 2
            if (f == y).all() or (f == 1 - y).all():
                hits.append((name, "".join(map(str, mask))))
    return hits, len(big), [int((lab == k).sum()) for k in big]


if __name__ == "__main__":
    for n in map(int, sys.argv[1:]):
        r, S = scan(n)
        lab = r["inv"][3]
        out = best_features(S, lab, n)
        print(f"\n=== N={n}")
        if out is None:
            print("  no hay exactamente 2 componentes no triviales; omitido")
            continue
        hits, nb, sizes = out
        print(f"  componentes no triviales: {nb}, tamaños {sizes}")
        print(f"  invariantes lineales mod 2 que coinciden: {len(hits)}")
        for h in hits[:12]:
            print("   ", h)
        # reparto de cada órbita D4 entre las dos componentes
        from build_graph import sym_images
        index = {tuple(s): i for i, s in enumerate(S)}
        seen, splits = set(), []
        big = [k for k in np.unique(lab) if (lab == k).sum() > 1]
        for i, s in enumerate(S):
            if i in seen:
                continue
            orb = {index[tuple(im)] for im in sym_images(s)}
            seen |= orb
            splits.append(tuple(sorted(int((lab[list(orb)] == k).sum()) for k in big)))
        from collections import Counter
        print("  reparto de cada órbita entre las 2 componentes:", dict(Counter(splits)))
