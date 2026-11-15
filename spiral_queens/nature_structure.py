import numpy as np
from spiral_queens import SpiralQueens
sq = SpiralQueens(7); N=7
print("sol[j]  ->  buscamos m,b tal que sol[j] = (m*j+b) mod N\n")
for sol,d,i,dr in sq.init_sols_label():
    s = [int(x) for x in sol]
    # deducir pendiente m = (sol[1]-sol[0]) mod N y verificar linealidad
    m = (s[1]-s[0]) % N
    b = s[0] % N
    lin = all((m*j+b)%N == s[j] for j in range(N))
    tag = "s1" if dr==1 else "s2"
    print(f"d={d} a={i} {tag}: {s}   m={m} b={b}  {'LINEAL ✓' if lin else 'no lineal'}")
