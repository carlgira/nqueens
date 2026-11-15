"""
Comprueba la prediccion de modular.py sobre un checkpoint (sirve a mitad de calculo):
¿que clases de recta modular usan las ondas minimas de N?

  python prediccion.py 13 [--ck data/checkpoints/waves_N13_nodeg_muestra200s0.jsonl]

Para N primo la prediccion es: clases = (N-3)/4 si N = 3 mod 4, (N-1)/4 si N = 1 mod 4,
todas usadas; para N = 1 mod 4 una de ellas es la especial {s : s^2 = -1} (tamano 2).
"""
import argparse, json, os
from collections import Counter, defaultdict
from modular import line_key, degenerate, predicted_classes, units, shortest
from analisis import isprime, slope

HERE = os.path.dirname(os.path.abspath(__file__))


def slope_class(key, N):
    """pendientes s = v/u mod N de la clase (para mostrarla como {s, -s, 1/s, -1/s})."""
    u, v = key
    s = v * pow(u, -1, N) % N
    return sorted({s, (-s) % N, pow(s, -1, N), (-pow(s, -1, N)) % N})


def main(N, ck):
    rows = {}
    for line in open(ck):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[d['clase']] = d
    print(f"N={N}: {len(rows)} clases calculadas en {os.path.basename(ck)}")
    K = Counter(len(d['sup_k']) for d in rows.values())
    ex = Counter(len(d['sup_k']) for d in rows.values() if d['exacto'])
    print(f"   ondas minimas: {dict(sorted(K.items()))}   (exactas: {dict(sorted(ex.items()))})")
    use, harm, eu = Counter(), defaultdict(Counter), defaultdict(Counter)
    recipes = defaultdict(Counter)
    for d in rows.values():
        cls = Counter()
        for k in d['sup_k']:
            key = line_key(tuple(k), N)
            use[key] += 1
            a, b, _ = slope(*shortest(tuple(k), N))
            eu[key][f'{a}:{b}'] += 1
            cls[key] += 1
        recipes[len(d['sup_k'])][tuple(sorted(cls.values(), reverse=True))] += 1
    posibles = {line_key((u, v), N) for u in range(N) for v in range(N)
                if (u, v) != (0, 0) and not degenerate((u, v), N)}
    n = sum(use.values())
    pred = predicted_classes(N) if isprime(N) else None
    print(f"   clases posibles (no degeneradas): {len(posibles)}   prediccion formula: {pred}   usadas: {len(use)}")
    for key in sorted(posibles, key=lambda k: -use[k]):
        sc = slope_class(key, N) if isprime(N) else ''
        esp = '  <- especial s^2 = -1' if isprime(N) and len(sc) == 2 else ''
        print(f"     clase {key}  pendientes {sc}: {100*use[key]/max(n,1):5.1f}% de las ondas{esp}"
              f"   dir. euclideas {dict(eu[key].most_common())}")
    print("   recetas (ondas por clase, ordenadas):")
    for k in sorted(recipes):
        print(f"     {k} ondas: {dict(recipes[k].most_common(6))}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('N', type=int)
    ap.add_argument('--ck')
    a = ap.parse_args()
    ck = a.ck or os.path.join(HERE, 'data', 'checkpoints', f'waves_N{a.N}_nodeg_muestra200s0.jsonl')
    main(a.N, ck)
