# Fraction Queens

Leer una permutación como los dígitos en base `N` de un número de `[0,1)`:

```
x(sigma) = 0.sigma(0) sigma(1) ... sigma(N-1)   en base N   =  M / N^N
```

y preguntar: **como números**, ¿se distinguen las soluciones de N-reinas de
las permutaciones que no lo son?

## Respuesta corta

En promedio **no**, en la cola **sí, y de forma brutal**.

- Todas las estadísticas medias coinciden con las de un racional cualquiera:
  longitud de la fracción continua, constante de Khinchin, media de `x`.
- Pero el conjunto de soluciones contiene un puñado de números que son
  **casi racionales de denominador diminuto**, y ninguna muestra del mismo
  tamaño de permutaciones al azar produce nada parecido.
- Esos números son exactamente las soluciones **mod-N / Sturmianas**
  (paso constante o dos pasos), las de [[sturmian-queens]]. La aritmética
  ve la misma frontera que ya salió en `wave_queens/` y `diag_waves/`.

## 1. La curva (`plots.py` → `fig_curve.png`)

Los `Q(N)` números ordenados forman una **S** respecto de la diagonal: las
soluciones huyen de los extremos de `[0,1)`.

| N | media sol | media perm | sd sol | sd perm | KS |
|---|---|---|---|---|---|
| 8 | 0.500000 | 0.500000 | 0.20205 | 0.28272 | 0.168 |
| 10 | 0.500000 | 0.499946 | 0.25552 | 0.28523 | 0.082 |
| 12 | 0.500000 | 0.500598 | 0.23251 | 0.28591 | 0.100 |

La media es **exactamente** 1/2 porque el conjunto de soluciones es cerrado
bajo `sigma -> N-1-sigma` (reflejo del tablero), que en el número es
`x -> 1-x`. La desviación es siempre menor: la banda del primer dígito está
sesgada al centro (para N=12: 3.5% en las bandas extremas contra 11.5% en las
centrales; para N=6 las bandas extremas están **vacías**).

En la densidad se ven las bandas de segundo nivel vacías: dentro de la banda
`sigma(0)=j` faltan las subbandas `j-1, j, j+1` porque `|sigma(1)-sigma(0)|>1`.
Es la restricción de las diagonales leída como **huecos en la expansión
decimal**.

## 2. El plano de la transpuesta (`fig_plane.png`)

`(x(sigma), x(sigma^-1))` = número del tablero contra número del tablero
transpuesto. Las permutaciones llenan la cuadrícula; las soluciones dejan
agujeros a toda escala (el zoom del centro sigue estando lleno de huecos).

## 3. Fracción continua: aquí está la señal (`cf.py` → `fig_cf.png`)

Para un racional `p/q` al azar la teoría dice: longitud de la FC
`~ 0.8427 ln q` (Heilbronn), media geométrica de los cocientes parciales
`-> 2.6854` (Khinchin), y ningún cociente domina. Definimos

```
L = len(FC) / (0.8427 ln q)      ~ 1   si es genérico
C = ln(max a_i) / ln q           ~ 0   si es genérico
```

`C` es la fracción de la información del número que se lleva **un solo**
cociente parcial, es decir, cómo de cerca está de un racional simple.

| N | L sol | L perm | C medio sol | C medio perm | **C max sol** | **C max perm** |
|---|---|---|---|---|---|---|
| 10 | 1.040 | 1.031 | 0.179 | 0.172 | **0.795** | 0.481 |
| 11 | 1.036 | 1.027 | 0.152 | 0.151 | **0.584** | 0.407 |
| 12 | 1.028 | 1.021 | 0.143 | 0.143 | **0.829** | 0.455 |

Las medias son iguales; las colas no. Y las soluciones de la cola son
identificables a simple vista:

```
N=12  C=0.829  sigma=[5,11,4,10,3,9,2,8,1,7,0,6]   pasos {5,6}
N=11  C=0.584  sigma=[0,5,10,4,9,3,8,2,7,1,6]      paso  {5}
N=10  C=0.795  sigma=[4,9,3,8,2,7,1,6,0,5]         pasos {4,5}
```

es decir, **la familia mod-N / Sturmiana**. Sus números son:

```
N=10   x ~ 40/81      |x - p/q| = 6.2e-12   (N^-N = 1.0e-10)
N=12   x ~ 60/121     |x - p/q| = 5.6e-15   (N^-N = 1.1e-13)
N=11   x ~ 59/1200    |x - p/q| = 1.4e-13   (N^-N = 3.5e-12)
N=13   x ~ 83/2016    |x - p/q| = 1.2e-16   (N^-N = 5.7e-13)
```

`81 = (N-1)^2`, `121 = (N-1)^2`, `1200 = (N+1)(N-1)^2`, `2016 = (N+1)(N-1)^2`.
Doce dígitos en base 12 que caben en `60/121`: el error es **más pequeño que
el último dígito**.

De dónde sale `(N-1)^2`: si los dígitos fueran una progresión aritmética
infinita `a + ck`,

```
sum_k (a+ck) N^-(k+1) = a/(N-1) + c/(N-1)^2
```

exactamente. Los `mod N` (los wraps) son la corrección, y a `N` dígitos vista
todavía cabe en `(N+1)(N-1)^2`. **Medido** (`law.py`): eso pasa exactamente
para `c` en `{(N-1)/2, (N+1)/2, N-1}` con `a` en `{0, c, N-1}` — 7 pares
`(a,c)` si `N` es impar, 2 si es par, verificado para `N = 5..40`.

## 4. Lo honesto: la cola no es de las soluciones, es de la aritmética

Toda progresión aritmética `sigma(k) = (a+ck) mod N` tiene `C` alto, **sea o
no solución** (`law.py`):

| N | C medio aritmética | C max aritmética | C medio al azar | C max al azar |
|---|---|---|---|---|
| 12 | 0.293 | 0.839 | 0.136 | 0.253 |
| 20 | 0.316 | 0.902 | 0.081 | 0.175 |
| 30 | 0.313 | 0.934 | 0.053 | 0.109 |

Lo que hace especial a las soluciones no es que sus números sean raros, sino
que **son el único sitio donde una progresión aritmética aparece con
probabilidad no despreciable**:

| N | Q(N) | aritméticas y solución | % en soluciones | % en permutaciones | enriquecimiento |
|---|---|---|---|---|---|
| 7 | 40 | 28 | 70.0% | 8.3e-01% | 84x |
| 11 | 2680 | 88 | 3.28% | 2.8e-04% | 1.2e4 x |
| 13 | 73712 | 130 | 0.176% | 2.5e-06% | 7.0e4 x |

(Y sale 0 para `N = 6,8,9,10,12`: paso constante exige `gcd(N,6)=1`, la misma
condición de [[sturmian-queens]]. En esos `N` la cola la ocupan las de **dos
pasos**, que la tabla de arriba no cuenta.)

## 5. Escala (`scaling.py` → `fig_scaling.png`)

`C` para `N = 8..60`, usando las tres construcciones cerradas de
`wave_queens/constructions.py`:

| | N=20 | N=40 | N=60 |
|---|---|---|---|
| mejor progresión aritmética | 0.902 | 0.951 | 0.967 |
| construcción clásica transpuesta | 0.807 | 0.950 | 0.934 |
| construcción clásica | 0.489 | 0.500 | 0.435 |
| solución genérica | 0.081 | 0.036 | 0.029 |
| permutación al azar | 0.081 | 0.034 | 0.027 |

La separación **crece** con N: lo genérico tiende a 0 y lo algebraico a 1.
Las FC lo dicen igual de claro: la permutación al azar tiene
`0.8427 N ln N` cocientes parciales (207 medidos, 207 predichos en N=60);
la construcción clásica, entre 10 y 40.

## 6. Lo que no distingue nada

Medido y negativo, para que no se vuelva a intentar:

- **Media geométrica de los cocientes parciales**: la *mediana* coincide
  (N=12: 2.619 sol vs 2.646 perm, ambas en Khinchin 2.685). La *media* sí
  difiere (3.57 vs 2.80) pero sólo porque un puñado de cocientes gigantes de
  la cola la arrastra: no es una propiedad del grueso.
- **Longitud de la FC**: `L = 1.03` vs `1.02`. Nada.
- **Huecos entre números consecutivos**: los dos conjuntos están igual de
  lejos de Poisson — agrupados por las bandas de dígitos, que son de la base
  N y no de las reinas. Las soluciones tienen algo más de cola (huecos >10x
  el medio: 1.5% vs 0.7% en N=12), que es el mismo sesgo del apartado 1.
- **Simplificación de `M/N^N`**: depende sólo de los dígitos finales y de los
  factores primos de N; artefacto de la base.
- **Dimensión de caja** del conjunto de fracciones: es `log Q(N)/(N log N)`,
  ya conocida de [[fractal-queens-tensor]] — la representación fraccional no
  añade nada ahí, sólo la hace literal (las cajas de lado `N^-d` *son* los
  prefijos del árbol de backtracking).

## Ficheros

- `frac.py` — soluciones, permutaciones y la fracción (`M/N^N`, FC exacta).
- `stats.py` — momentos, dígitos, dimensiones, racional reducido.
- `cf.py` — las medidas `L`, `C`, `G` y la tabla soluciones vs permutaciones.
- `law.py` — la ley del denominador y el enriquecimiento aritmético.
- `complexity.py` — `q*` = menor denominador que ya reproduce los N dígitos.
- `scaling.py` — `N = 8..60` con las construcciones clásicas.
- `near.py` — a qué racional simple se pega cada solución.
- `plots.py` — las cinco figuras.
- `out_*.txt` — las salidas completas de `stats.py`, `cf.py`, `law.py`.
