"""
Permute column sources onto fixed column indices for SuperQueens init_sols candidates.

For permutation perm, v[i] = sol[perm[i]]: column i receives the row that was in column perm[i].

Dos modos:
- Completo: todas las permutaciones de S_n (o filtradas por nº de índices con perm[i]!=i).
- Restringido (--restricted-shuffle): eliges un subconjunto S de columnas con
  |S| en [k_min, k_max] (por defecto 2 .. n//2); fuera de S se fuerza perm[i]=i.
  Solo se reordena qué fila llega a cada columna de S usando una biyección S→S.
  Así el espacio es sum_k C(n,k)·k! en lugar de n!.

Solo estadísticas por consola (sin gráficos); matplotlib queda en backend Agg al importar.
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from super_queens import SuperQueens  # noqa: E402
import nqueens  # noqa: E402


def apply_column_perm(sol, perm):
    """Column i gets the row from source column perm[i]."""
    return [int(sol[perm[i]]) for i in range(len(perm))]


def moved_indices_count(perm):
    """Count of positions i where perm[i] != i (non-fixed points in one-line form)."""
    return sum(1 for i, p in enumerate(perm) if p != i)


def perm_allowed(perm, n, include_identity, k_min, k_max):
    if not include_identity and all(perm[i] == i for i in range(n)):
        return False
    k = moved_indices_count(perm)
    if k_min is not None and k < k_min:
        return False
    if k_max is not None and k > k_max:
        return False
    return True


def iter_restricted_column_permutations_unique(n, k_min, k_max, include_identity):
    """
    Solo columnas en un subconjunto S (|S| entre k_min y k_max) se reasignan con una biyección S→S;
    fuera de S, perm[i]=i. La misma permutación global puede salir de varios S; se deduplica con `seen`.
    """
    seen = set()
    identity = tuple(range(n))
    for k in range(k_min, k_max + 1):
        if not 0 <= k <= n:
            continue
        for S in itertools.combinations(range(n), k):
            slist = list(S)
            for tup in itertools.permutations(slist):
                perm = list(range(n))
                for col, src in zip(slist, tup):
                    perm[col] = src
                p = tuple(perm)
                if not include_identity and p == identity:
                    continue
                if p in seen:
                    continue
                seen.add(p)
                yield p


def count_restricted_raw_generations(n, k_min, k_max):
    """Intentos brutos (S, tup) sin deduplicar; sum_k C(n,k)·k!."""
    s = 0
    for k in range(k_min, k_max + 1):
        if not 0 <= k <= n:
            continue
        s += math.comb(n, k) * math.factorial(k)
    return s


def sol_tuple(sol):
    if hasattr(sol, "tolist"):
        sol = sol.tolist()
    return tuple(int(x) for x in sol)


def run_experiment(
    n,
    include_identity=False,
    k_min=None,
    k_max=None,
    verbose_perm=False,
    restricted_shuffle=False,
):
    sq = SuperQueens(n)
    candidates = sq.init_sols()
    base_init = {sol_tuple(s) for s in candidates}

    monitor = nqueens.n_queens(n)
    all_sols = [tuple(s) for s in monitor.all_solutions]
    all_set = set(all_sols)

    union_shuffled = set()
    per_perm_hits = []

    if restricted_shuffle:
        rk_min = 2 if k_min is None else k_min
        rk_max = n // 2 if k_max is None else k_max
        if rk_min > rk_max:
            rk_min, rk_max = rk_max, rk_min
        rk_min = max(0, min(rk_min, n))
        rk_max = max(0, min(rk_max, n))
        perms_iter = iter_restricted_column_permutations_unique(
            n, rk_min, rk_max, include_identity
        )
        total_perm = count_restricted_raw_generations(n, rk_min, rk_max)
        mode = "restringido (solo subconjuntos S con |S| en [k_min,k_max], resto identidad)"
        subset_range = (rk_min, rk_max)
    else:
        perms_iter = itertools.permutations(range(n))
        total_perm = math.factorial(n)
        mode = "completo S_n"
        subset_range = None

    used_perm = 0
    perms_with_any_valid = 0
    perms_with_new_vs_init = 0
    total_valid_pairs = 0
    total_valid_pairs_new_vs_init = 0

    max_distinct_valid_per_shuffle = 0
    min_distinct_valid_per_shuffle = None
    sum_distinct_valid_per_shuffle = 0
    max_distinct_new_outside_init_per_shuffle = 0
    sum_distinct_new_outside_init_per_shuffle = 0
    perms_with_at_least_2_distinct_valid = 0
    perms_with_at_least_2_distinct_new = 0
    hist_distinct_valid = {}

    for perm in perms_iter:
        if not restricted_shuffle:
            if not perm_allowed(perm, n, include_identity, k_min, k_max):
                continue
        used_perm += 1
        found_here = set()
        for sol in candidates:
            v = apply_column_perm(sol, perm)
            if sq.validate(v):
                t = tuple(v)
                union_shuffled.add(t)
                found_here.add(t)
                total_valid_pairs += 1
                if t not in base_init:
                    total_valid_pairs_new_vs_init += 1
        nv = len(found_here)
        n_new = sum(1 for t in found_here if t not in base_init)
        if nv > 0:
            perms_with_any_valid += 1
            max_distinct_valid_per_shuffle = max(max_distinct_valid_per_shuffle, nv)
            if min_distinct_valid_per_shuffle is None:
                min_distinct_valid_per_shuffle = nv
            else:
                min_distinct_valid_per_shuffle = min(min_distinct_valid_per_shuffle, nv)
            sum_distinct_valid_per_shuffle += nv
            sum_distinct_new_outside_init_per_shuffle += n_new
            max_distinct_new_outside_init_per_shuffle = max(
                max_distinct_new_outside_init_per_shuffle, n_new
            )
            if nv >= 2:
                perms_with_at_least_2_distinct_valid += 1
            if n_new >= 2:
                perms_with_at_least_2_distinct_new += 1
            hist_distinct_valid[nv] = hist_distinct_valid.get(nv, 0) + 1
            if any(t not in base_init for t in found_here):
                perms_with_new_vs_init += 1
        if verbose_perm:
            per_perm_hits.append((perm, nv, found_here))

    base_valid = {t for t in base_init if sq.validate(list(t))}

    not_in_base = union_shuffled - base_init
    not_in_base_valid_init = union_shuffled - base_valid
    covered_of_all = union_shuffled & all_set
    outside_all = union_shuffled - all_set

    pa = perms_with_any_valid
    mean_dv = sum_distinct_valid_per_shuffle / pa if pa else 0.0
    mean_new = sum_distinct_new_outside_init_per_shuffle / pa if pa else 0.0

    return {
        "n": n,
        "mode": mode,
        "restricted_shuffle": restricted_shuffle,
        "subset_range": subset_range,
        "num_candidates": len(candidates),
        "base_init_count": len(base_init),
        "base_valid_count": len(base_valid),
        "permutations_total": total_perm,
        "permutations_used": used_perm,
        "union_shuffled_count": len(union_shuffled),
        "not_in_base_init": len(not_in_base),
        "not_in_base_valid": len(not_in_base_valid_init),
        "all_solutions_count": len(all_set),
        "union_intersect_all": len(covered_of_all),
        "union_minus_all": len(outside_all),
        "missed_by_union": len(all_set - union_shuffled),
        "verbose_rows": per_perm_hits,
        "perms_with_any_valid": perms_with_any_valid,
        "perms_with_new_vs_init": perms_with_new_vs_init,
        "total_valid_pairs": total_valid_pairs,
        "total_valid_pairs_new_vs_init": total_valid_pairs_new_vs_init,
        "max_distinct_valid_per_shuffle": max_distinct_valid_per_shuffle,
        "min_distinct_valid_per_shuffle": min_distinct_valid_per_shuffle,
        "mean_distinct_valid_per_shuffle": mean_dv,
        "max_distinct_new_outside_init_per_shuffle": max_distinct_new_outside_init_per_shuffle,
        "mean_distinct_new_outside_init_per_shuffle": mean_new,
        "perms_with_at_least_2_distinct_valid": perms_with_at_least_2_distinct_valid,
        "perms_with_at_least_2_distinct_new": perms_with_at_least_2_distinct_new,
        "hist_distinct_valid": hist_distinct_valid,
    }


def main():
    p = argparse.ArgumentParser(
        description="Column permutations on SuperQueens.init_sols() and n-queens validation."
    )
    p.add_argument("--n", type=int, default=7, help="Board size (default 7)")
    p.add_argument(
        "--restricted-shuffle",
        action="store_true",
        help=(
            "Solo mezclar un subconjunto S de columnas (|S| entre k-min y k-max); "
            "el resto perm[i]=i. Por defecto k-min=2, k-max=N//2. Reduce n! a ~sum C(N,k)k!."
        ),
    )
    p.add_argument(
        "--include-identity",
        action="store_true",
        help="Include the identity permutation (default: skip)",
    )
    p.add_argument(
        "--k-min",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Modo completo: mín. índices con perm[i]!=i. "
            "Modo --restricted-shuffle: mín. tamaño |S| (default 2)."
        ),
    )
    p.add_argument(
        "--k-max",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Modo completo: máx. índices movidos. "
            "Modo --restricted-shuffle: máx. |S| (default N//2)."
        ),
    )
    p.add_argument(
        "--verbose-perm",
        action="store_true",
        help="Listar las 20 permutaciones con más soluciones válidas (solo texto)",
    )
    args = p.parse_args()

    stats = run_experiment(
        args.n,
        include_identity=args.include_identity,
        k_min=args.k_min,
        k_max=args.k_max,
        verbose_perm=args.verbose_perm,
        restricted_shuffle=args.restricted_shuffle,
    )

    print("--- Estadísticas (n-reinas, shuffle de columnas sobre init_sols) ---")
    print("n:", stats["n"])
    print("modo:", stats["mode"])
    if stats["restricted_shuffle"] and stats["subset_range"]:
        lo, hi = stats["subset_range"]
        print(
            "  (tamaños |S| de columnas que participan en la biyección; el resto queda orden fijo)"
        )
        print(f"  rango |S|: [{lo}, {hi}]")
        print(
            "  intentos brutos (S,tup) sin deduplicar (cota trabajo):",
            stats["permutations_total"],
        )
        print("  permutaciones globales únicas tras deduplicar:", stats["permutations_used"])
    else:
        print("permutaciones en S_n (factorial n):", stats["permutations_total"])
        print("permutaciones tras filtro por índices movidos:", stats["permutations_used"])
    print("candidatos init_sols:", stats["num_candidates"])
    print("vectores distintos en init_sols:", stats["base_init_count"])
    print("válidos sin shuffle (subset init_sols):", stats["base_valid_count"])
    print()
    print("Permutaciones que generan ≥1 solución válida (algún init_sol):", stats["perms_with_any_valid"])
    print("De esas, permutaciones con ≥1 solución distinta fuera de init_sols inicial:", stats["perms_with_new_vs_init"])
    print("(pares perm×candidato con resultado válido):", stats["total_valid_pairs"])
    print("de los cuales resultado ∉ init_sols inicial:", stats["total_valid_pairs_new_vs_init"])
    print()
    print("--- Por shuffle (soluciones = vectores válidos distintos en ese shuffle) ---")
    print(
        "máx. soluciones distintas en un solo shuffle:",
        stats["max_distinct_valid_per_shuffle"],
    )
    if stats["min_distinct_valid_per_shuffle"] is not None:
        print(
            "mín. (solo shuffles con ≥1 solución):",
            stats["min_distinct_valid_per_shuffle"],
        )
    print(
        "media de soluciones distintas por shuffle (solo shuffles con ≥1):",
        f"{stats['mean_distinct_valid_per_shuffle']:.4f}",
    )
    print(
        "shuffles con ≥2 soluciones distintas:",
        stats["perms_with_at_least_2_distinct_valid"],
    )
    print()
    print("--- Soluciones fuera del conjunto init_sols inicial (por shuffle) ---")
    print(
        "máx. soluciones distintas fuera de init en un solo shuffle:",
        stats["max_distinct_new_outside_init_per_shuffle"],
    )
    print(
        "media de soluciones distintas fuera de init por shuffle (solo shuffles con ≥1 válida):",
        f"{stats['mean_distinct_new_outside_init_per_shuffle']:.4f}",
    )
    print(
        "shuffles con ≥2 soluciones distintas fuera de init:",
        stats["perms_with_at_least_2_distinct_new"],
    )
    if stats["hist_distinct_valid"]:
        hist = sorted(stats["hist_distinct_valid"].items())
        parts = [f"{k} sols:{v} shuffles" for k, v in hist]
        print("histograma (nº soluciones distintas → cuántos shuffles):", "; ".join(parts))
    print()
    print("Soluciones válidas distintas (unión sobre permutaciones):", stats["union_shuffled_count"])
    print("de ellas, no estaban en el conjunto init_sols inicial:", stats["not_in_base_init"])
    print("de ellas, no estaban en los válidos sin shuffle:", stats["not_in_base_valid"])
    print()
    print("Todas las soluciones del solver:", stats["all_solutions_count"])
    print("intersección unión shuffle ∩ solver:", stats["union_intersect_all"])
    print("unión shuffle fuera del solver:", stats["union_minus_all"])
    print("soluciones del solver no alcanzadas por la unión:", stats["missed_by_union"])

    if args.verbose_perm and stats["verbose_rows"]:
        top = sorted(stats["verbose_rows"], key=lambda x: -x[1])[:20]
        print("\nTop 20 permutaciones por nº de soluciones válidas en esa perm:")
        for perm, cnt, _ in top:
            print(f"  válidas={cnt} perm={perm}")


if __name__ == "__main__":
    main()
