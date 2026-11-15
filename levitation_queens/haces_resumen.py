"""
Resumen de data/haces_base_p{p}_{n}_{m}.txt (salida de ./haces): por pareja de pendientes,
soluciones base, orbitas de traslacion = base*(p-1)/(n(n-1)m), tipo de pendientes, |H| y clase real.
Uso: ../venv/bin/python haces_resumen.py p n m [muestra]
"""
import sys
from collections import defaultdict, Counter
from haces import clase
from tipos import nombre
from clase_m import subgrupo, E

p, n, m = map(int, sys.argv[1:4]); muestra = int(sys.argv[4]) if len(sys.argv) > 4 else 40
por = defaultdict(list)
for ln in open(f"data/haces_base_p{p}_{n}_{m}.txt"):
    v = list(map(int, ln.split())); por[(v[0], v[1])].append(v[2:])
fam = defaultdict(lambda: [0, 0, Counter()])
for (a, b), Ts in por.items():
    H = len(subgrupo(E(a, b, p), p))
    key = (len(Ts), "/".join(sorted([nombre(a, p), nombre(b, p)])), H)
    f = fam[key]; f[0] += 1; f[1] += len(Ts) * (p - 1) // (n * (n - 1) * m)
    for T in Ts[:max(1, muestra // 8)]: f[2][clase(T, p, n + m)] += 1
tot = 0
for (base, nom, H), (np_, orb, cl) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
    tot += orb
    print(f"  pendientes [{nom}] |H|={H:2d} bloques={(p-1)//H:2d}: {np_:2d} parejas x {base} base -> {orb} orbitas; clase real (muestra): {dict(cl)}")
print(f"p={p} haz {n}+{m}: {tot} orbitas de traslacion (con multiplicidad si una solucion admite dos haces)")
