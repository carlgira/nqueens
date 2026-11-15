# Spiral N-Queens

Núcleo extraído de `super_queens.py`: la parte que genera y dibuja las
soluciones de las N reinas (para N primo) como **espirales logarítmicas**
construidas con fracciones del número áureo `phi`.

## Idea

Se monta una rejilla **polar** de `N × N` puntos:

- `N` ángulos equiespaciados (rayos): `a[i] = 2πi/N`
- `N` radios que crecen según una espiral logarítmica basada en `phi` (anillos)

Si se traza una espiral logarítmica con constante `c = phi^(1/(d·π))` y un
desfase angular `a`, la espiral pasa por **exactamente un punto de cada radio**,
y esos `N` puntos forman una **solución válida de las N reinas**.

Recorriendo:

- las fracciones de `phi` → parámetro `d = 1 … ⌊N/2⌋-1`
- los `N` desfases angulares
- las 2 direcciones de giro de la espiral

se obtienen todas las soluciones "espirales". Para **N = 7**: `2 × 7 × 2 = 28`.

## Archivos

- `spiral_queens.py` — clase `SpiralQueens` (rejilla, `get_sol`, `init_sols_label`, `validate`).
- `plot_solutions.py` — dibuja todas las soluciones en un grid con matplotlib.

## Uso

```bash
python spiral_queens.py        # imprime las 28 soluciones de N=7 y valida
python plot_solutions.py       # grid 4×7 con las 28 soluciones (N=7)
python plot_solutions.py 11    # otro primo
```

Genera `spiral_queens_n<N>.png`. En el grid: **azul = espiral 1**,
**rojo = espiral 2**; los puntos negros son las reinas sobre la espiral.
Las de `d=1` dan una vuelta; las de `d=2`, dos vueltas.

## Versión 3D

Cada anillo `j` se inclina a su propio plano rotándolo `tilt_j = k·j·π/N`
alrededor del eje X. La espiral 2D se convierte en una curva 3D:

```
x = r·cos(θ)
y = r·sin(θ)·cos(tilt)
z = r·sin(θ)·sin(tilt)
```

- `plot_solutions_3d.py` — grid de las 28 en 3D, o una sola en grande.
- `spin_3d.py` — GIF rotatorio de una solución (aquí se ve la torsión).

```bash
python plot_solutions_3d.py            # grid 4x7 en 3D (N=7)
python plot_solutions_3d.py 7 one 14   # una sola en grande
python plot_solutions_3d.py 7 one 14 2 # ... con k=2
python spin_3d.py 7 14 1               # GIF rotatorio
```

### Qué curva es (la pregunta divertida)

Con `u = ln(r)` resulta que **tanto el ángulo azimutal como la inclinación son
lineales en `u`**:

- azimut:      `θ(u)   = α·u + θ0`   (porque `θ = log_c(r) + cte`)
- inclinación: `tilt(u) = β·u + φ0`  (porque `ln r[j]` es lineal en `j`)

Así que la curva es una **espiral logarítmica con dos frecuencias angulares
acopladas** (radio exponencial `r = e^u`, una velocidad `α` para girar y otra
`β` para inclinarse):

```
x = e^u · cos(α u + θ0)
y = e^u · sin(α u + θ0) · cos(β u + φ0)
z = e^u · sin(α u + θ0) · sin(β u + φ0)
```

Es un pariente de la *concho-espiral* / espiral cónica: cuando `β/α` es racional
(que es el caso, todo vive en una rejilla N-aria) la curva se "cierra" en los
nudos que se ven en las figuras.
