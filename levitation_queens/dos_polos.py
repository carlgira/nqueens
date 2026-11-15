"""
Clase 3 (HAZ 2+1 sobre i, -i): la relacion entre los dos cruces, en una tabla.

Cada fila r (salvo los dos cruces r1, r2) tiene DOS direcciones:
  turno visto desde el cruce 1:  i = log((r-r1)/d) mod k
  turno visto desde el cruce 2:  j = log((r-r2)/d) mod k        d = r2-r1, k = (p-1)/4
Tabla k x k: casilla (i, j) = cuantas filas tienen esas dos direcciones. La tabla es la misma para
todas las soluciones de un p (son los "numeros ciclotomicos" de orden k).
Solucion = elegir columnas X (la espiral A1 se queda con ellas enteras) y renglones Y (A2 con ellos
enteros) de modo que todas las casillas donde se cruzan esten VACIAS. B se queda con el resto.

Uso: ../venv/bin/python dos_polos.py            -> cuenta N(p) y comprueba contra los volcados
     ../venv/bin/python dos_polos.py fig 17 29  -> fig_dos_polos.png
"""
import sys
from collections import defaultdict, Counter
from itertools import product
import numpy as np
from espirales_log import raiz_primitiva


def tabla(p):
    g = raiz_primitiva(p); k = (p - 1) // 4
    log = {pow(g, t, p): t for t in range(p - 1)}
    M = np.zeros((k, k), int)
    for x in range(2, p):
        M[log[x] % k, log[x - 1] % k] += 1
    return M, log, k


def cuenta(p):
    """Pares (X, Y) no vacios con todas las casillas X x Y vacias."""
    M, _, k = tabla(p)
    n = 0
    for X in range(1, 2 ** k):
        libres = [j for j in range(k) if all(M[i, j] == 0 for i in range(k) if X >> i & 1)]
        n += 2 ** len(libres) - 1
    return n


def comprueba(p):
    """Todas las clase 3 del volcado: A1 = columnas enteras, A2 = renglones enteros, cruces vacios."""
    M, log, k = tabla(p)
    inv = lambda x: pow(x, p - 2, p)
    a = next(a for a in range(2, p) if a * a % p == p - 1)
    ok = tot = 0
    for linea in open(f"data/clases_toro_p{p}.txt"):
        v = list(map(int, linea.split()))
        if v[0] != 3 or v[1] != 0:
            continue
        T = v[1:]
        rect = defaultdict(list)
        for r in range(p):
            for s in (a, p - a):
                rect[(s, (T[r] - s * r) % p)].append(r)
        gr = sorted(rect.items(), key=lambda kv: -len(kv[1]))[:3]
        sl = Counter(s for (s, _), _ in gr)
        par = next(s for s in sl if sl[s] == 2)
        A = [(ac, f) for ac, f in gr if ac[0] == par]
        (sb, cb), _ = next((ac, f) for ac, f in gr if ac[0] != par)
        r1, r2 = [(cb - c) * inv((s - sb) % p) % p for (s, c), _ in A]
        d = inv((r2 - r1) % p)
        d1 = lambda r: log[(r - r1) * d % p] % k
        d2 = lambda r: log[(r - r2) * d % p] % k
        X = {d1(r) for r in A[0][1] if r != r1}
        Y = {d2(r) for r in A[1][1] if r != r2}
        bien = ({r for r in range(p) if r != r1 and d1(r) in X} == set(A[0][1]) - {r1}
                and {r for r in range(p) if r != r2 and d2(r) in Y} == set(A[1][1]) - {r2}
                and all(M[i, j] == 0 for i, j in product(X, Y)))
        tot += 1; ok += bien
    return ok, tot


def figura(primos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(primos), figsize=(5.6 * len(primos), 6), squeeze=False)
    for ax, p in zip(axs[0], primos):
        M, _, k = tabla(p)
        # ejemplo: el par (X, Y) valido con mas casillas ocupadas repartidas
        mejor = max(((X, Y) for X in range(1, 2 ** k) for Y in range(1, 2 ** k)
                     if all(M[i, j] == 0 for i in range(k) for j in range(k) if X >> i & 1 and Y >> j & 1)),
                    key=lambda xy: min(bin(xy[0]).count("1"), bin(xy[1]).count("1")) * 100
                    + bin(xy[0]).count("1") + bin(xy[1]).count("1"))
        X, Y = mejor
        for i in range(k):
            for j in range(k):
                enX, enY = X >> i & 1, Y >> j & 1
                col = "#c9c9c9" if enX and enY else "#9ecae1" if enX else "#fcae91" if enY else "white"
                ax.add_patch(plt.Rectangle((i, j), 1, 1, fc=col, ec="0.5", lw=0.8))
                if M[i, j]:
                    ax.text(i + .5, j + .5, str(M[i, j]), ha="center", va="center", fontsize=11)
        ax.set_xlim(0, k); ax.set_ylim(0, k); ax.set_aspect("equal")
        ax.set_xticks(np.arange(k) + .5); ax.set_xticklabels(range(k))
        ax.set_yticks(np.arange(k) + .5); ax.set_yticklabels(range(k))
        ax.set_xlabel("turno visto desde el cruce 1"); ax.set_ylabel("turno visto desde el cruce 2")
        ax.set_title(f"p={p}: {k} turnos, {cuenta(p)} elecciones validas\n"
                     "azul = columnas de la espiral 1, rojo = renglones de la espiral 2,\n"
                     "gris = donde se cruzan (tiene que estar vacio)", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig_dos_polos.png", dpi=120)
    print("-> fig_dos_polos.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fig":
        figura([int(x) for x in sys.argv[2:]] or [17, 29])
    else:
        for p in (13, 17, 29, 37, 41):
            print(p, "N(p) por la tabla =", cuenta(p))
        for p in (13, 17):
            print(p, "volcado: cumplen %d de %d" % comprueba(p))
            print(tabla(p)[0])
