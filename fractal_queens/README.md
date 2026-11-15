# Fractal Queens

Construir un fractal *que sea* una solución de N-reinas, no un dibujo bonito
al lado.

## El resultado exacto (verificado)

Sea `sigma (x) tau` la sustitución por bloques: en el tablero de `sigma`
(tamaño `n`), cada reina se reemplaza por una copia completa del tablero de
`tau` (tamaño `m`), y cada casilla vacía por un bloque vacío. El resultado es
un tablero `n*m` con `n*m` reinas — el producto de Kronecker de las matrices
de permutación.

> **`sigma (x) tau` es solución válida de N-reinas para TODA `sigma` válida
> si y sólo si `tau` es solución TOROIDAL.**

Comprobado por fuerza bruta para todas las `tau` de `m = 4..9` contra todas
las `sigma` de `n = 5,6,7`: cero contraejemplos en ambos sentidos
(`tensor.py`).

### Por qué

Con `x = i*m + j` y `rho(x) = sigma(i)*m + tau(j)`:

```
x - rho(x) = A*m + B      A = i - sigma(i)      B = j - tau(j)
```

Como `|B| <= m-1`, dos valores sólo pueden colisionar si `A - A' = ±1` y
`B - B' = ∓m`. Es decir: la única forma de romper una diagonal es que dos
diferencias de `tau` disten exactamente `m`. Que eso no ocurra equivale a que
`{j - tau(j)}` sea un sistema completo de residuos módulo `m` — la condición
toroidal. Idéntico para las sumas.

La condición **no depende de `sigma`**. Luego la torre `tau^(x)k` es válida
para todo `k`, y `sigma (x) tau^(x)k` también.

### El fractal

`tau^(x)k` es el k-ésimo paso de un IFS de `m` contracciones de razón `1/m`:

```
f_j(z) = ( z + (j, tau(j)) ) / m
```

El atractor es un polvo autosemejante en el cuadrado unidad con dimensión
`log m / log m = 1` (cumple la condición del conjunto abierto: los bloques no
se solapan). Sus proyecciones sobre los ejes son el intervalo completo (una
reina por fila y columna a toda escala) y sus proyecciones a ±45° son
inyectivas a toda escala — que es exactamente la condición de las diagonales
leída como propiedad del fractal.

`tau(j) = 3j mod 7` da la torre 7, 49, 343, 2401, 16807, ... (`fig_tensor_*`).

## Ficheros

- `tensor.py` — `kron`, `valid`, `toroidal`, enumeración.
- `render_tensor.py` — `fig_tensor_tower.png`, `fig_tensor_zoom.png`.
- `sturmian_plane.py` / `plot_sturmian.py` — plano `(alpha,beta)` coloreado
  por conflictos. **Ojo:** a `N` fijo esto es un arreglo de rectas de Farey,
  no un fractal; el refinamiento ocurre al crecer `N`, no al hacer zoom
  (`fig_sturmian_cells.png`: 36.6% → 31.9% → 19.0% del área para N=7,13,31).

---

# Espirales: la solución codificada, no dibujada

Tres niveles de codificación. El tablero no se ve en ninguna.

## 1. Geométrica — `newton_spiral.py`

Las reinas son las **raíces** de un polinomio en ℂ:
`z_k = (k−c)/N + i(σ(k)−c)/N`, `c=(N−1)/2`. Se dibuja Newton relajado

```
F(z) = z − a·p(z)/p'(z) = z − a / Σ_k 1/(z − z_k)
```

Con `a=1` cada raíz es superatractora (multiplicador 0) y las cuencas son
lóbulos concéntricos. Con `a` **complejo** el multiplicador en cada reina pasa
a ser `1−a ∈ ℂ`: la cuenca entra girando y toda la frontera se enrosca en
espirales logarítmicas de paso `arg(1−a)/log|1−a|`.

Decodificación: los atractores del sistema *son* las reinas.
(`fig_newton_spiral.png`, a = 1, 1+0.45i, 1+0.95i)

## 2. Simbólica — `spiral_ifs.py`

Un IFS de N semejanzas directas, una por reina, con los **dos índices
diagonales** (que viven módulo `2N−1`):

```
d⁻(k) = k − σ(k) + N − 1        d⁺(k) = k + σ(k)

f_k(z) = r·e^{2πi d⁺(k)/(2N−1)} · z  +  e^{2πi d⁻(k)/(2N−1)}
```

`d⁻` dice **dónde** va cada copia, `d⁺` dice **cuánto gira**. El atractor es un
dragón de brazos en espiral; `r = N^(−1/dim)` fija la dimensión.

> **σ es solución ⟺ el IFS tiene N posiciones distintas y N giros distintos.**

Si dos reinas comparten diagonal, dos copias se solapan o giran igual: el
fractal pierde un brazo. La solución es la configuración de máxima asimetría.
Verificado en `fig_spiral_ifs.png`: dos soluciones de N=7 dan 7/7 y 7/7 y
fractales distintos; una no-solución da 4/7 posiciones y se le ve el colapso.

**Intento fallido, y por qué.** La primera versión usaba `f_k(z) = r·e^{2πiσ(k)/N} z + e^{2πik/N}`
— giro de la columna, posición de la fila. No codifica nada: si giros y
traslaciones son ambos raíces N-ésimas de la unidad, el IFS conmuta con Z_N y
el atractor tiene simetría C_N sea cual sea σ. Dos soluciones distintas daban
la misma rosácea. Hay que salir del módulo N: por eso las diagonales, que
viven módulo 2N−1.

`fig_spiral_hero.png`: N=13, σ(j)=5j mod 13, dim 1.9.

## 3. Aritmética — pendiente

La más profunda. La familia Sturmiana da soluciones vía un número de rotación
α. En dinámica compleja `z → e^{2πiα}z + z²` tiene disco de Siegel si α es
diofántico, y la frontera del disco está hecha de espirales cuya combinatoria
*es* la fracción continua de α. Mismo número haciendo los dos trabajos: resolver
el tablero y curvar el fractal. Falta elegir el α noble dentro del intervalo
Sturmiano válido para cada N y renderizar.

---

# Todas las soluciones en un solo fractal

Tres ejes independientes, los tres verificados (`mixing.py`, `moran.py`).

## Eje 1 — PROFUNDIDAD: una solución distinta por escala

Torre de Moran heterogénea. `P₁ = τ₁` (cualquier solución), `P_{k+1} = P_k ⊗ τ_{k+1}`
con `τ_{k+1}` toroidal y de **tamaño libre**. La prueba del teorema sólo usaba
`|B| ≤ m−1`, que no depende del tamaño exterior, así que los factores pueden ir
cambiando de talla.

```
8 × 5 × 7 × 11 × 13 = 40040     válida = True
```

Una sola solución de 40040 reinas en la que cada profundidad de zoom guarda
otra solución: N=8 (no toroidal) arriba del todo, luego 2j mod 5, 3j mod 7,
3j mod 11, 5j mod 13. El coarse-grain de cada ventana devuelve exactamente
`τ_k` — comprobado con `assert` en los 5 niveles (`fig_moran_zoom.png`).

## Eje 2 — RAMA: distinta solución en cada bloque

Si cada bloque `i` usa su propia `τ^(i)`, la colisión pide
`A_i − A_{i'} = ±1` **y** `B − B' = ∓m` con `B ∈ D_{τ^(i)}`, `B' ∈ D_{τ^(i')}`.
Dentro de un bloque lo prohíbe la toroidalidad; entre bloques hay que exigirlo:

> **τ y τ' son mezclables ⟺ `D_τ = D_{τ'}` y `S_τ = S_{τ'}`.**

Predicado verificado contra fuerza bruta: 400/400 para m=5 y m=7. Mezclar al
azar **nunca** funciona (0/400). Pero las clases de compatibilidad no son
triviales: tienen **exactamente 2 elementos**, y la pareja es

```
τ(j) = c·j mod m      y      τ⁻¹(j) = c⁻¹·j mod m
```

o sea la solución y su **transpuesta**. Mezclar dentro de la clase vale siempre:
300/300 (m=5), 840/840 (m=7), 4080/4080 (m=11).

## Consecuencia: el fractal no guarda una solución, guarda un árbol de ellas

Cada bloque de cada nivel elige libre e independientemente entre `τ` y `τ⁻¹`.
Para la torre `8 × 5 × 7 × 11` eso son `8 + 40 + 280 = 328` bits:

```
2^328 ≈ 10^99 soluciones distintas de N=3080 sobre el MISMO esqueleto fractal
```

200/200 torres aleatorias válidas. El esqueleto (dónde hay bloques) es común;
lo que cambia es la quiralidad de cada bloque.

## Eje 3 — pendiente: todas las Q(N) de un N fijo

Lo anterior apila soluciones de tamaños distintos. La otra lectura — meter las
92 soluciones de N=8, o las 352 de N=9, en un solo objeto — es el **árbol de
backtracking** dibujado radialmente: profundidad = reinas colocadas, cada hoja
superviviente una solución, las ramas muertas se cortan a su profundidad. Ese
árbol es un Cantor con dimensión `lim log(supervivientes a d)/(d log N)`.

## Ficheros nuevos

- `mixing.py` — `kron_mixed`, `predict_mixed`, clases de compatibilidad.
- `moran.py` — `tower`, `window`, `moran_points` (versión espiral).
- `render_moran.py` — `fig_moran_zoom.png`, `fig_moran_spiral.png`.

---

# Eje 3 — el árbol: todas las Q(N) de un N fijo en un objeto

`tree.py`, `render_tree.py`. Cada colocación parcial sin conflictos es un nodo;
radio = profundidad, ángulo repartido proporcionalmente entre los hijos. Las
ramas que mueren antes de llegar a N se pintan apagadas; las que llegan, en
plasma; las hojas son las soluciones.

```
N=8    2057 nodos     92 soluciones    73% del árbol es poda
N=10  35539 nodos    724 soluciones    86% del árbol es poda
```

## Lo que se mide

**Perfil.** `W_d` = nodos sin conflicto a profundidad `d`. Crece, culmina y
colapsa. El pico está siempre en `d* ≈ 0.7 N` — el momento en que la búsqueda
es más ancha, y a partir de ahí las diagonales matan más de lo que el tablero
deja crecer:

```
  N    d*   d*/N     W_max     nodos     Q(N)   Q/W_max
  8     5   0.625       568      2057       92    0.162
 10     7   0.700      9632     35539      724    0.075
 12     9   0.750    222720    856189    14200    0.064
```

**Dimensión.** Escribiendo cada solución como `0.σ(0)σ(1)…σ(N−1)` en base `N`,
el conjunto de soluciones son `Q(N)` celdas de tamaño `N^-N` dentro de `[0,1)`,
o sea un Cantor de dimensión de caja

```
dim = log Q(N) / (N log N)
```

Con la asintótica de Simkin `Q(N) ≈ (N e^{−α})^N`, `α ≈ 1.944`, eso tiende a
`1 − α/ln N`: la **codimensión del Cantor de las N-reinas es la constante de
Simkin dividida por `ln N`**. La convergencia es logarítmica y por tanto
lentísima — a N=12 el valor real (0.321) y la asíntota (0.218) todavía van
separados y ambos subiendo (`fig_tree_profile.png`). No es una comprobación de
la fórmula, sólo la misma tendencia.

`fig_tree_radial.png` (N=8), `fig_tree_radial_n10.png` (N=10).
