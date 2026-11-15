# levitation_queens — ¿cuántas ondas hacen falta para levitar una solución?

Idea: un levitador acústico crea una **red N×N de trampas** (pozos de potencial
en el aire). Encima se superponen **K ondas planas** que hacen más profundos
unos pozos que otros. Si las N trampas más profundas son exactamente las N
reinas, las N partículas caen en una solución.

**Pregunta**: para cada solución, ¿cuál es el **mínimo K** y con qué parámetros?

---

## El modelo

```
f(i, j) = Σₖ Aₖ · cos( 2π (k₀·i + k₁·j) / N + φₖ )        reinas = las N casillas con f > umbral
```

- `k = (k₀, k₁)` es el vector de onda en *ciclos por N casillas* a lo largo de
  (filas, columnas). Una onda = una dirección + una separación entre franjas.
  Longitud de onda en casillas: `N / |k|`.
- El umbral (la constante, `k = 0`) es libre y **no cuenta** como onda.
- Solo importan los valores **en las casillas**, porque la red de trampas ya fija
  dónde pueden estar las partículas. Entre casillas el campo puede tener picos más
  altos, y da igual.
- Se exige un margen: `f(reina) ≥ umbral + δ` y `f(resto) ≤ umbral − δ`, con
  `|a|, |b| ≤ 1` por onda (`a = A cos φ`, `b = −A sin φ`) y `δ = 10⁻³`. El resultado
  **no cambia** con `δ = 10⁻²` ni con `5·10⁻²` (comprobado en N=8 y N=9), así que no
  son soluciones colgadas de un hilo numérico.

## Cómo se calcula el mínimo (exacto)

Con el conjunto de ondas fijado, "¿separan las reinas del resto?" es un **LP**
(lineal en `a, b, umbral`). Elegir el conjunto más pequeño es combinatorio, así
que se plantea como **MILP**: una variable binaria `zₖ` por onda posible
("la uso o no"), `|aₖ|, |bₖ| ≤ zₖ`, y se minimiza `Σ zₖ`. HiGHS lo resuelve con
certificado de optimalidad. Luego, con el soporte ya fijado, otro LP maximiza el
margen para dar parámetros robustos.

- Ondas candidatas: todas las `k ∈ Z_N²` (conmensuradas con la red), una por par
  `±k`: unas N²/2.
- Solo se resuelve un representante por clase de simetría (las 8 del cuadrado). Los
  demás se obtienen transformando `k' = L·k` y reajustando fases con un LP. **Cada
  fila del fichero se verifica**: se reconstruye el campo y se comprueba que sus N
  casillas más brillantes son las reinas.

### Dos variantes

| fichero | ondas permitidas | por qué |
|---|---|---|
| `data/waves_N{N}.json` | toda `k` de la rejilla | el problema matemático puro |
| `data/waves_N{N}_fisico.json` | solo longitud de onda ≥ 2 casillas (`\|k\| ≤ N/2`) | si la red de trampas se hace con la misma frecuencia, la separación entre trampas es λ₀/2, y ninguna onda de esa frecuencia tiene franjas más juntas que λ₀ = 2 casillas |

### "Una onda libre"

Además se comprueba si **una sola onda con k no entero** basta. Con una onda el
conjunto iluminado es un **arco de fases**, así que el test es exacto para cada k:
¿ocupan las reinas un arco contiguo del círculo de fases? Se barre `k` en una
rejilla fina (paso 1/8). Un `null` significa que no se encontró ninguna, no que sea
imposible.

---

## Resultados

Ondas mínimas (rejilla completa) → número de soluciones:

| N | soluciones | 1 | 2 | 3 | 4 | 5 | 6 | imposible | 1 onda libre |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2 | · | 2 | · | · | · | · | · | 2 |
| 5 | 10 | 10 | · | · | · | · | · | · | 10 |
| 6 | 4 | · | 4 | · | · | · | · | · | 4 |
| 7 | 40 | 28 | 4 | · | · | 8 | · | · | 32 |
| 8 | 92 | · | 4 | · | 88 | · | · | · | 4 |
| 9 | 352 | · | 16 | 240 | 48 | 48 | · | · | 4 |
| 10 | 724 | · | 16 | 12 | 448 | 248 | · | · | 28 |

Todos los mínimos son exactos (certificados). En N=10 dos clases necesitaron más de 15 min de MILP; se cerraron con `refinar.py` (confirmado: 5).

Variante física (λ ≥ 2 casillas):

| N | soluciones | 1 | 2 | 3 | 4 | 5 | 6 | imposible | 1 onda libre |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 2 | · | · | · | · | · | · | 2 | 2 |
| 5 | 10 | 10 | · | · | · | · | · | · | 10 |
| 6 | 4 | · | · | 4 | · | · | · | · | 4 |
| 7 | 40 | 28 | · | 4 | · | 8 | · | · | 32 |
| 8 | 92 | · | · | 4 | 24 | 64 | · | · | 4 |
| 9 | 352 | · | 16 | 232 | 56 | 40 | 8 | · | 4 |
| 10 | 724 | · | · | 28 | 360 | 240 | 96 | · | 16 |

Lo que se lee:

- **1 onda en la rejilla = soluciones toroidales/modulares**, exactamente: 10 (N=5),
  28 (N=7). Es `j = a·i mod N`, la construcción de Pólya.
- **El mínimo real es mucho menor que el de "ir sumando las ondas más fuertes de la
  FFT"**. Para N=8 aquel método daba 3–9; el mínimo exacto es 2 o 4.
- **Ninguna solución de N=6, 8, 9, 10 cabe en 1 onda de la rejilla** (no hay
  toroidales), y el grueso necesita 4–5. El máximo observado es 5 (6 en la física).
- **Onda libre → pocas ondas de rejilla**: toda solución que se pinta con una onda
  libre necesita como mucho 3 en la rejilla (N=9: sus 4 necesitan 3; N=10: 16 con 2 y
  12 con 3). Y al revés, en N=8 y N=10 las de 2 ondas son todas "de onda libre".
- **Las soluciones de 2 ondas en N=8 son las mismas 4 que se pintan con una sola onda
  libre**, por ejemplo `#13 = [2,4,1,7,0,6,3,5]` con `k = (3, −1.875)`. Un `k` no entero
  sobre la red equivale a dos ondas enteras. Esto **amplía** lo de `wave_queens/`
  (Sturmian): allí N=8 quedaba sin cubrir con la rotación 1D, y aquí una onda plana
  2D (un conjunto de "corte y proyección") sí cubre 4 soluciones de N=8. Pero es
  frágil: su hueco relativo es solo 0.0375, frente a 0.17 para la versión de 2
  ondas enteras.
- **N=4 y N=6 no tienen soluciones toroidales**, pero una onda libre las pinta todas.
  En la variante física, N=4 es **imposible** con ondas de la rejilla: no quedan
  suficientes direcciones con λ ≥ 2.
- Pedir realismo físico (λ ≥ 2) encarece: N=8 pasa de {2, 4} a {3, 4, 5} ondas.

## Relación con lo que ya había (para no repetir)

| proyecto | qué hace | diferencia con esto |
|---|---|---|
| `diag_waves/` | onda → **testigo**: la potencia del espectro cuenta conflictos; `wave2d.py` es la DFT 2D completa (inyectiva) | análisis: del tablero al espectro. Aquí es **síntesis**: cuál es el espectro más pobre que *produce* el tablero |
| `spiral_queens/prony*.py` | mínimo nº de modos `p` para representar **exactamente** la secuencia 1D `z_j` de una solución | exacto y 1D. Aquí es 2D y por **umbral** (solo importan los picos), que es lo que ve una partícula. Distribuciones distintas: N=7 Prony {1:28, 4:12}, aquí {1:28, 2:4, 5:8} |
| `wave_queens/rotation.py` (Sturmian) | rotación 1D `rank(frac(αk+β))` | "una onda libre" aquí es la versión 2D, y cubre soluciones de N=8 que la 1D no alcanzaba |

## Uso

```bash
python min_waves.py 4 5 6 7 8 9 10            # rejilla completa  -> data/waves_N*.json
python min_waves.py 4 5 6 7 8 9 10 --fisico   # lambda >= 2       -> data/waves_N*_fisico.json
python consultar.py 8                          # resumen
python consultar.py 8 13 --png                 # ficha de la solucion 13 y figura del campo
python consultar.py 8 --sol 2 4 1 7 0 6 3 5    # buscar por la permutacion
```

Formato de cada solución en el JSON:

```json
{"id": 13, "solucion": [2,4,1,7,0,6,3,5], "clase": 9,
 "ondas_min": 2, "exacto": true, "cota_inferior": 2,
 "umbral": 1.0, "margen": 0.1716,
 "ondas": [{"k": [3,-2], "amplitud": 1.0824, "fase_rad": -2.7489,
            "longitud_onda_casillas": 2.2188, "direccion_grados": 123.69},
           {"k": [4,1],  "amplitud": 0.4483, "fase_rad": -1.1781,
            "longitud_onda_casillas": 1.9403, "direccion_grados": 75.96}],
 "una_onda_libre": [3.0, -1.875]}
```

`solucion[i]` es la columna de la reina de la fila `i`. `direccion_grados` es el
ángulo del vector de onda (0° = a lo largo de las columnas, 90° = a lo largo de
las filas).

---

## Transferencia a N mayores (sin MILP): `transfer.py`

Se reutilizan los **conjuntos de vectores k** hallados en N=4..10 (230 distintos
salvo simetría) y, para cada solución del N nuevo, un LP reajusta solo amplitudes,
fases y umbral. Se prueban dos formas de trasladar k (mismo k / misma longitud de
onda en casillas) y las 8 orientaciones. Resultado = **cota superior** del mínimo.

**N=11** (2,5 min): 440 de 2680 soluciones levitadas (16 %).

| ondas | 1 | 2 | 3 | 4 | 5 | ninguno sirve |
|---|---|---|---|---|---|---|
| soluciones | 88 | 16 | 60 | 120 | 156 | 2240 |

- Las 88 de 1 onda son las toroidales, heredadas de N=5 y N=7: `k = (1, a)` sirve
  para cualquier N con los mcd adecuados.
- "Misma longitud de onda" (k escalado, no entero) **no aportó ninguna**: lo que se
  hereda es el número de ciclos por tablero, no la escala física.
- El 84 % sin cubrir no significa que necesite más de 5 ondas, solo que sus ondas
  no están entre las heredadas. Para saber su mínimo real haría falta el MILP de
  N=11 (horas).

**N=12** (15 min): 244 de 14200 soluciones levitadas (1,7 %).

| ondas | 1 | 2 | 3 | 4 | 5 | ninguno sirve |
|---|---|---|---|---|---|---|
| soluciones | · | · | 8 | 54 | 182 | 13956 |

- La hipótesis "en N=11 funcionó mal por ser primo" **no se sostiene**: sin contar las
  toroidales, N=11 cubre 352/2592 = 13,6 % y N=12 solo el 1,7 %. N compuesto va peor.
- No hay de 1 onda (N=12 no tiene toroidales) ni de 2. Casi todo viene de conjuntos
  de 4–5 ondas de N=10 (208 de 244), y otra vez solo con "mismo k".
- Lectura: el repertorio heredado es fijo (230 conjuntos) y el número de soluciones
  crece ~5× por cada N, así que la cobertura cae. Las soluciones de N grande usan
  combinaciones de k que en N ≤ 10 no existen (hay más frecuencias disponibles).

`python plot_transfer.py 11|12` → `fig_transfer_N{N}.png`. Datos: `data/transfer_N{N}.json`.

---

## ¿Se repiten, se completan, se multiplican?

### Repetición (`data/waves_N*.json`, salvo simetría)

| N | soluciones | conjuntos de ondas distintos | el más usado |
|---|---|---|---|
| 7 | 40 | 5 | 16 soluciones |
| 8 | 92 | 23 | 8 |
| 9 | 352 | 30 | 36 |
| 10 | 724 | 169 | 16 |

Dentro de cada N se repiten mucho (N=9: 352 soluciones con solo 30 conjuntos).
**Entre Ns distintos, casi nunca**: solo `{(1,−2)}` (N=5 y 7, la toroidal) y
`{(1,−3),(3,2)}` (N=7 y 9). Ojo: el MILP devuelve *un* conjunto óptimo de muchos
posibles, así que esto subestima la repetición real.

### Completar: K−1 ondas de un N menor + 1 onda nueva (`completar.py`)

Para cada clase de N' con mínimo exacto K se prueban todos los conjuntos de K−1
ondas heredados (mismo k, 8 orientaciones) más cada onda extra. **Control**: el
mismo número de conjuntos de K−1 ondas elegidos al azar.

| N' | K | clases | completables heredando | con conjuntos al azar |
|---|---|---|---|---|
| 9 | 3 | 32 | **32** | 8 |
| 9 | 5 | 6 | 2 | 4 |
| 10 | 4 | 56 | **48** | 10 |
| 10 | 5 | 31 | **9** | 0 |
| 8 | 4 | 11 | — (no hay conjuntos de 3 ondas con N<8) | — |

**Sí: las soluciones de N mayor "crecen" desde las de N menor añadiendo una onda.**
En N=9, *todas* las de 3 ondas son un conjunto de 2 ondas heredado + 1; en N=10,
el 86 % de las de 4 ondas. Al azar, 4–5 veces menos. (K=2 tiene muy pocas clases
para concluir nada.)

### Múltiplos: producto σ⊗τ (`multiples.py`)

`ρ(i·m+j) = σ(i)·m + τ(j)` es solución de N=n·m para toda σ si τ es toroidal
(`fractal_queens/`). ¿Levitan el producto las ondas de σ y τ?

- **Tal cual, no**: 0 de 1816 productos (4×5, 5×5, 6×5, 7×5, 4×7, 5×7, 8×5), ni con
  σ con el mismo k + τ con k·n, ni añadiendo los términos cruzados `k_σ ± n·k_τ`.
  Razón: el producto es una **escalera de bloques** y una onda plana es suave; su
  fase deriva dentro de cada bloque de m×m.
- **Con los alias `k_σ + n·t`** (t ∈ Z_m², lo que hace falta para dibujar bloques)
  sí, y el MILP dentro de ese conjunto da pocas ondas: **N=25 (5×5) con 4, N=20
  (4×5) con 6**, N=30 (6×5) con ≤ 7–8. O sea, multiplicar cuesta unas pocas ondas
  más que la suma de las dos, no m² veces más.

### ¿El salto es solo entre N seguidos? (`saltos.py`)

**A — por N de origen** (clases completables con base de K−1 ondas de cada N1):

| N' | K | clases | N1=4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| 9 | 3 | 32 | 2 | – | 10 | 17 | 22 | |
| 10 | 3 | 2 | 0 | – | 2 | 2 | 2 | 2 |
| 10 | 4 | 56 | – | – | – | – | – | 48 |
| 10 | 5 | 31 | – | – | – | – | 9 | 3 |

(– = ese N1 no tiene conjuntos de K−1 ondas.) **No es solo consecutivo**: cuando hay
bases de varios N todos aportan, incluso N=4 → 9, y N=8 → 10 gana a N=9 → 10. Lo
que parece consecutivo (K=4 de N=10 solo desde N=9) es disponibilidad: los únicos
conjuntos de 3 ondas están en N=9.

**B — de ≤10 a 11** (base de 1–3 ondas heredada + 1 onda nueva, 5 min):

- Levita 336 soluciones con ≤ 4 ondas (108 con 2, 80 con 3, 148 con 4). Orígenes:
  N=9 124, N=7 108, N=5 104, N=10 76, N=8 44 (una clase puede venir de varios).
- **No amplía la cobertura**: las 336 ya estaban entre las 440 del transfer directo.
  Lo que hace es **abaratarlas** (p. ej. 20 pasan a 2 ondas, 88 de 5 bajan a 3–4).
  Mejor de ambos: 1: 88 · 2: 20 · 3: 124 · 4: 120 · 5: 88 · sin cubrir 2240.
- Para alcanzar las 2240 restantes haría falta partir de bases de 4 ondas (+1 → 5)
  o añadir 2 ondas; estimado ~40 min con `--maxb 4`.

**C — N=12, 13, 14 sobre muestra aleatoria** (base de 1–3 ondas de cada N1 + 1 onda;
para N1 divisor, también k escalado ×N/N1; ~30 min):

| N | muestra | levitadas | desde N1 que aportan | divisores (4, 6, 4×3, 6×2 / 7, 7×2) |
|---|---|---|---|---|
| 11 (completo) | 2680 | 12,5 % | 5, 7, 8, 9, 10 | — |
| 12 | 400 | 1,5 % | 9 (6), 10 (3) | **0** |
| 13 | 300 | 0,7 % | todos (las mismas 2) | — |
| 14 | 300 | 0 % | ninguno | **0** |

- **Ser divisor no ayuda**, ni con el mismo k ni con k escalado: la geometría de un
  tablero pequeño repetido no dibuja soluciones de uno grande.
- La cobertura **cae con N** (12,5 → 1,5 → 0,7 → 0 %) y lo que funciona viene de los
  N más cercanos y con más bases (9, 10).
- Por qué los múltiplos no aparecen aquí: el producto σ⊗τ necesita τ **toroidal**
  (m ∈ {5, 7, 11, …}) y σ de n ≥ 4, así que los primeros N compuestos por producto
  son 20, 25, 28, 30, 35. 12 = 4×3 = 6×2 y 14 = 2×7 no se pueden factorizar así
  (no hay soluciones de 2 ni 3 reinas). Donde sí existe el producto (`multiples.py`)
  levita con pocas ondas añadiendo los alias.

---

## N=20 y N=25 (`grandes.py`)

Ya no se enumeran: 200 soluciones "normales" por N con min-conflicts aleatorio (no es
muestreo uniforme), todos los productos σ⊗τ (4×5 → 20, 5×5 → 100), y MILP sobre
todas las frecuencias con 10 min de límite en 3 productos + 3 normales.

| | N=20 (4×5) | N=25 (5×5) |
|---|---|---|
| productos | 20 | 100 (todos toroidales, ninguno lineal ni de 1 onda) |
| ondas de un producto (alias, MILP) | ≤ 6 | 4 (exacto en alias) |
| ondas de un producto (MILP completo, 10 min) | 5, 6, 7 | **3, 3, 3** (≥ 2: se comprobó que 1 no basta) |
| ondas de una normal (MILP completo, 10 min) | 10, 11, 12 | 14, 16, 39 |
| normales que son productos | 0/200 | 0/200 |
| normales levitadas con ondas de productos | 0/200 | 0/200 |
| normales levitadas con ondas heredadas de N≤10 | 0/200 | 0/200 |

Todas las cifras del MILP completo son **cotas superiores** (no cerró en 10 min; las
cotas inferiores que da son triviales, 1–2). El 39 es claramente un incumbente malo.

- **Los productos son soluciones "baratas"**: 3 ondas en N=25 frente a ≥ 14 para las
  normales; 5–7 frente a 10–12 en N=20. La multiplicación sí fabrica soluciones
  espectralmente simples de N grande.
- Pero son **rarísimas** (20 de ~3.9·10¹⁰ en N=20; 100 de ~2.2·10¹⁵ en N=25) y sus
  ondas **no sirven para nada más**: 0 de 200 normales. Tampoco las heredadas.
- 5⊗5 da soluciones toroidales de 25 que **no** son lineales `a·i+b mod 25`: toroidal
  no implica 1 onda cuando N es compuesto. Necesitan 2–3.

---

## De dónde salen los ángulos: la regla de las 4 proyecciones

**Observación.** Las direcciones usadas son pendientes racionales pequeñas (1:2 la más
frecuente, luego 2:3, 1:3, 1:4, 3:4), las mismas en todos los N, y **nunca** 0°/90°
(fila/columna), casi nunca 45°.

**Hipótesis.** Una onda `k = (u, v)` "mide" el tablero a lo largo de las cuatro
direcciones de ataque con las proyecciones

```
u,  v,  u+v,  u−v      (mod N)        fila, columna, diagonal ↘, diagonal ↗
```

Si alguna vale 0, la onda es constante a lo largo de esa línea: la ilumina entera y
no puede separar reinas dentro de ella. Versión fuerte: lo mejor es que las cuatro
sean coprimas con N (la condición de Pólya de las toroidales, generalizada).

**Comprobación con los datos** (ondas usadas vs. todas las k posibles):

| N | usadas con alguna proyección 0 | al azar |
|---|---|---|
| 4 | 0 % | 78 % |
| 5 | 0 % | 67 % |
| 6 | 0 % | 58 % |
| 7 | 0 % | 50 % |
| 8 | 8,9 % | 46 % |
| 9 | 1,4 % | 40 % |
| 10 | 3,9 % | 37 % |

N primo: 100 % de las usadas tienen las 4 proyecciones coprimas (azar 33–50 %).
N=9: 100 % tienen 3 de 4, el máximo posible (de u, v, u±v una es múltiplo de 3).

**Prueba de fuego** (`python min_waves.py ... --nodeg`): restringir el MILP a ondas
sin proyecciones cero da **exactamente los mismos mínimos, solución por solución**,
en N=4..10 (724/724 en N=10, todas certificadas), y es mucho más rápido: N=9 17 s
vs 200 s; N=10 ~3 min vs 33 min + 2 h de `refinar.py` para las 2 clases difíciles. Las pocas ondas "degeneradas" de N=8 y N=10
tenían alternativas equivalentes no degeneradas.

**Regla generativa para otros N**: candidatas = `{k : u, v, u+v, u−v ≢ 0 (mod N)}`,
ordenadas preferentemente por cuántas proyecciones son coprimas con N.

**N=11** con `--nodeg` (3 h 25 min, 10 núcleos, `--time-limit 900`):

| ondas | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| soluciones | 88 | 20 | 188 | 236 | 1660 | 488 |

- Exactas todas las de 1–5 ondas y 40 de las de 6. Las otras **448 de 6 ondas son
  cota** (cota inferior 4 en 304, 3 en 96, 5 en 48). Preguntar directamente
  "¿≤ 5 ondas?" a una de ellas (clase 46) tampoco cerró en 25 min.
- 1 onda = las 88 toroidales, como siempre. El grueso necesita 5.
- La transferencia desde N ≤ 10 (`transfer.py`) acertaba el mínimo en 244 de sus
  440; en las otras 196 sobraba 1–2 ondas.
- Direcciones: 1:2 sigue primera (18 %), pero aparecen con fuerza pendientes con 5
  (2:5 13 %, 3:5 11 %, 1:5 9 %) que en N ≤ 10 eran raras: el repertorio de ángulos
  crece con N (denominadores hasta ~N/2).

### Parar y reanudar

`min_waves.py` guarda cada clase resuelta en `data/checkpoints/waves_N{N}[_fisico][_nodeg].jsonl`
(una línea por clase, con `fsync`). Si se para —Ctrl-C, `kill`, apagar el equipo— basta
relanzar **el mismo comando** y sólo calcula las pendientes:

```bash
nohup sh -c '../venv/bin/python -u min_waves.py 11 --nodeg --time-limit 900 2>&1 | grep --line-buffered -v Highs' > run_N11_nodeg.log 2>&1 &
tail -f run_N11_nodeg.log                      # progreso: [hechas/total] clase, ondas, segundos
pkill -f "min_waves.py 11"; pkill -f multiprocessing   # parar
```

Una clase a medio calcular al parar se pierde (se recalcula entera). El JSON final
`data/waves_N11_nodeg.json` se escribe cuando están todas.

---

## Análisis global N=4..11 (`analisis.py`, `plot_analisis.py` → `fig_analisis.png`)

- **Máximo de ondas = ceil(N/2) para N = 8, 9, 10, 11** (en N=11 hay 40 soluciones con 6
  certificado). Falla en N ≤ 7: N=7 tiene 8 soluciones con 5 > 4; N=5, 6 quedan por
  debajo. La media/moda van ~0,5–1 por debajo y oscilan (N=9 tiene moda 3).
- **Orden de Farey máximo = floor(N/2) exacto en todo N** — pero es en parte forzado:
  con el alias más corto |u|,|v| ≤ N/2. Lo no trivial es que el repertorio se usa
  entero: cada N estrena las direcciones con denominador floor(N/2) (N=10, 11: 1:5,
  2:5, 3:5, 4:5).
- **Separación de franjas casi constante**: mediana |k|/N ≈ 0,40–0,45 → franjas cada
  ~2,2–2,5 casillas sea cual sea N. La escala la pone la casilla, no el tablero.
- **Las ondas de una solución son "independientes"**: pares perpendiculares y tríadas
  k₁ ± k₂ = k₃ al nivel del azar (N=9: tríadas *evitadas*, 3 % vs 17 %).
- **Lo que agrupa es la cercanía en N, no la paridad ni la primalidad**: parecido medio
  N↔N+1 0,70, N↔N+2 0,51; par-par 0,43, impar-impar 0,36, par-impar 0,49. Los primos
  (5, 7, 11) son los *menos* parecidos entre sí (0,24) porque están lejos.
- **La aritmética de N sí fija las proyecciones**: primos 100 % con las 4 coprimas;
  N=9 100 % con 3 de 4; pares 55 % con 2 y 22 % con 3.

---

## Los ángulos viven módulo N (`modular.py` → `fig_modular.png`)

El ángulo euclídeo a:b es un artefacto del alias más corto. El objeto natural es la
**recta modular** `{t·k mod N}`. Para N primo una recta es una pendiente
`s = v/u mod N`; las cuatro direcciones de ataque son `s = 0, ∞, 1, −1`, y las 8
simetrías del tablero actúan como `s → −s, 1/s, −1/s`. Toda onda es

```
k = t · (1, s)   (mod N)        s = clase de recta,  t = armónico (1 … N/2)
```

y lo que se ve como "ángulo nuevo" (2:5, 3:5, 1:5…) es un **armónico plegado** de una
recta vieja. Por ejemplo, en N=11: 3·(1,2) = (3,6) ≡ (3,−5) → 3:5 es la recta de 1:2.

| N | direcciones euclídeas | **clases de recta usadas** | posibles | fórmula (primos) |
|---|---|---|---|---|
| 5 | 1 | 1 | 1 | 1 |
| 7 | 3 | **1** | 1 | 1 |
| 11 | 9 | **2** | 2 | 2 |
| 8, 9, 10 | 6, 6, 11 | 4, 2, 6 (+ sub-rectas) | 4, 2, 6 | — |

- **Se usan todas las clases no degeneradas** en cada N (salvo N=6: 2 de 3), y dentro
  de cada clase **todos los armónicos** con frecuencias parecidas.
- **Fórmula para N primo**: clases = (N−3)/4 si N ≡ 3 (mod 4), (N−1)/4 si N ≡ 1 (mod 4)
  (el grupo de Klein {s, −s, 1/s, −1/s} sobre las N−3 pendientes permitidas; si N ≡ 1
  mod 4 existe `s² = −1`, una clase de tamaño 2, la "que rota 90° sobre sí misma").
  **Predicción**: N=13 → 3 clases ({2,11,7,6}, {3,10,9,4}, {5,8}), 17 → 4, 19 → 4, 23 → 5.
- **N compuesto** añade sub-rectas (k con mcd(k, N) > 1, periodo N/g): (2,4) en N=8 y
  10 (periodo N/2 — "un tablero de la mitad" dentro del grande), (2,2), (0,1).
- **Cada solución es una receta de clases**: en N=11 las de 5 ondas son 3+2, 2+3, 1+4
  o 4+1 ondas de las dos rectas; las de 3, 2+1, 1+2 o 3+0.

### Por qué: el espectro de la solución

`S(k) = Σ_reinas e^{2πi k·x/N}` (su transformada de Fourier en Z_N²).

- **Proyección de fila o columna 0 ⇒ S(k) = 0 exactamente** para toda permutación
  (`Σᵢ e^{2πi v σ(i)/N} = 0`). Medido: ≤ 5·10⁻¹⁵ en todas. Es la demostración de que
  0°/90° nunca pueden servir. Proyección diagonal 0 ⇒ S pequeño (1,2–1,8 de media).
- **Las ondas elegidas son picos de S**: puesto mediano 2–4 entre 18–40 candidatas,
  percentil 0,72–0,84. No siempre los K mayores (por eso el "greedy" no es óptimo).

---

## Síntesis: ir directo a las soluciones (`generador.py`)

Hasta aquí todo era **análisis** (dada una solución, ¿cuántas ondas?). Al revés: elegir
ondas del vocabulario de la teoría (no degeneradas = clases de recta × armónicos),
sumarlas, dejar caer una partícula por fila en la casilla más brillante, y ver si sale
solución. Sin conocer las soluciones.

**A ciegas** (amplitudes y fases al azar): las ondas de la teoría aciertan ~2× más que
todas, pero casi sólo salen las toroidales (N=11: 88 + 1). Una suma aleatoria queda
dominada por una onda.

**Afinando las fuentes** (búsqueda local sobre fases/amplitudes y a veces cambiar una
onda, 2000 cadenas × 1500 pasos):

| N | K=2 | K=3 | K=ceil(N/2) | unión | tiempo |
|---|---|---|---|---|---|
| 8 | 20 | 30 | 50 | **56 = 61 % de Q** | ~15 s |
| 10 | 61 | 62 | 72 | **128 = 18 % de Q** | ~40 s |
| 12 | | 11 | 6 | | ~30 s |
| 13 | | 195 (130 tor.) | 157 (130 tor.) | | ~45 s |
| 16 | | **61** | 4 | | ~65 s |
| 20 | | **12** | 0 | | ~2 min |

- Encuentra soluciones de N=16 y N=20 en segundos **sin conocerlas**, usando sólo 3
  ondas de la teoría.
- **Pocas ondas se afinan mejor que muchas**: con K=ceil(N/2) el espacio de perillas es
  demasiado grande. Por eso el generador está sesgado hacia las soluciones
  espectralmente simples (las de pocas ondas) — no muestrea la masa de soluciones.
- La teoría da el **vocabulario** (qué ondas pueden servir), no la **frase** (qué
  combinación da cada solución): elegir la combinación sigue siendo el problema.

### ¿Ayuda la estructura de los primos al generador? (`primos.py`)

Las clases de recta salen exactamente como predice la fórmula también fuera de los
datos: 11 → 2, 13 → 3 (especial {5,8}), 17 → 4 (especial {4,13}, 4² ≡ −1), 19 → 4,
23 → 5. Comparando elegir las K ondas **al azar** del vocabulario contra **por receta**
(repartidas entre clases distintas), soluciones NO toroidales encontradas (2000 × 1500):

| N | K=3 azar / receta | K=5 azar / receta |
|---|---|---|
| 11 | 28 / 29 | 21 / **35** |
| 13 | 56 / 54 | 28 / 31 |
| 17 | **30** / 15 | **10** / 6 |
| 19 | 0 / 1 | 1 / 0 |
| 23 | 1 / 1 | 0 / 0 |

- **La receta no ayuda de forma consistente** (mejor en 11, peor en 17, empate en el
  resto). La estructura en clases describe *qué* ondas existen, pero repartirlas entre
  clases no es la regla que elige *cuáles* combinan.
- En primos grandes el generador casi no encuentra no toroidales (19, 23: 0–1), mientras
  que en compuestos parecidos sí (N=16: 61, N=20: 12 con K=3). Posible lectura: en N
  primo las soluciones "de pocas ondas" son casi todas toroidales; las demás necesitan
  muchas ondas y el afinado se pierde. No comprobado.

---

## Fuentes laterales en pares (reinterpretación física, sin recalcular)

Una "onda" del modelo no tiene por qué venir de una fuente en la cúpula: dos altavoces
**laterales** de la misma frecuencia, separados un ángulo θ en el anillo, interfieren y
dan `|p_a + p_b|² = cte + 2|p_a||p_b| cos(k·x + φ)` con `k = k_a − k_b`: exactamente un
término del modelo (la constante la absorbe el umbral). Dirección de franjas = bisectriz;
separación `d = λ / (2 sin(θ/2))`. Con λ = 2 casillas (la frecuencia de la red):
θ = 2·asin(1/d) → d = 5: 23°, d = 2,2: 54°, d = 1,56 (la más fina posible en N=11): 80°.

- **Ningún resultado cambia**: los mínimos, ángulos y clases sólo dependen de los
  vectores k, no de cómo se fabrica cada onda.
- **La variante `--fisico` queda obsoleta**: suponía haz de referencia vertical (franjas
  ≥ 2 casillas). Con pares laterales la franja mínima es λ/2 = 1 casilla, y el alias más
  corto de cualquier k tiene |k| ≤ N/√2, es decir, franjas ≥ 1,41 casillas. **Todas las
  ondas de la rejilla son realizables** y la variante física coincide con la completa.
- Usando una frecuencia ligeramente distinta por par, los pares no interfieren entre sí
  y sus franjas se suman sin términos cruzados, como asume el modelo.
- Coste: 2K altavoces + 4 para la red (N=11 típica, 5 ondas: 14 altavoces).

---

## ¿Bonitas vs. el resto? Cuántas representaciones mínimas tiene cada solución (`representaciones.py`)

Para cada clase de N=7, 8, 9 se prueban TODOS los conjuntos de K ondas (K = su mínimo) y se
cuentan los que la levitan (R), su **núcleo** (ondas presentes en todas las
representaciones) y cuántas ondas distintas aparecen en total.

| tipo | R | núcleo común | ondas usadas en total | ejemplos |
|---|---|---|---|---|
| toroidales (K=1) | (N−1)/2 | 0 | (N−1)/2: **todos los armónicos de una misma recta** | N=7: 28 sol |
| **2 ondas** | **1 — rígidas** | las 2 | 2 | N=7: 4, N=8: 4, N=9: 16 |
| 3 ondas, N=9 | 4 | **1 onda fija** + 1 de 6 intercambiables | 6 | 232 sol |
| 4–5 ondas | 16–162 | 0–1 | casi **todo el vocabulario** (22–24 de 24 en N=9) | el grueso |

- **Hay dos regímenes**: soluciones **estructuradas** (una familia de armónicos, una pareja
  única, o un núcleo fijo con pocas alternativas) y soluciones **amorfas** (sin núcleo:
  cualquier combinación de K ondas de casi todo el vocabulario vale; en N=7 las de 5
  ondas se levitan con el 20 % de TODOS los conjuntos de 5).
- Las de 2 ondas son **rígidas en todos los N medidos**: una única pareja posible.
- En proporción a los C(n, K) conjuntos posibles, R es parecido (0,1–1,5 %) salvo
  toroidales y N=7; lo que separa los regímenes no es R sino el **núcleo**: las
  estructuradas comparten ondas fijas, las amorfas no.

### Núcleo con N (`nucleo.py`, reanudable)

Núcleo = ondas presentes en todas las representaciones mínimas (MILP: prohibir la onda y
ver si aún hay K). Validado contra el recuento completo de N=7–9.

| N | 2 ondas | 3 ondas | 4 ondas | 5 ondas |
|---|---|---|---|---|
| 7 | rígidas | — | — | sin núcleo |
| 8 | rígidas | — | núcleo 1 en 80/88 | — |
| 9 | rígidas | núcleo 1 en 240/240 | núcleo 2 en 32/48 | sin núcleo |
| 10 | rígidas | sin núcleo (12) | **rígidas 24**, núcleo 2: 88, 1: 176, 0: 160 | 0: 120, ≥2: 8, sin resolver 120 |

- Aparecen **rígidas de 4 ondas en N=10**: la parte estructurada crece con N.
- No es escalera limpia: en N=10 las de 3 ondas son amorfas.
- **N=11** (`python nucleo.py 11 --kmax 4`, límite 120 s por prueba):
  2 ondas: rígidas 20/20 · **3 ondas: rígidas 80/188 (43 %)**, núcleo 2 en 72, sin núcleo 36 ·
  4 ondas: 32 sin núcleo, 8 con núcleo 1, 196 sin resolver (28 con núcleo ≥ 1) ·
  5 ondas (30 clases): ningún núcleo demostrado, casi todo sin resolver.

Rígidas por N (nº de soluciones): N=7: 4 (K=2) · N=8: 4 (K=2) · N=9: 16 (K=2) ·
N=10: 16 (K=2) + 24 (K=4) · N=11: 20 (K=2) + 80 (K=3). La parte estructurada crece con N
y sube de K, pero sin regla limpia (N=10 salta a K=4 sin pasar por 3; N=11 tiene K=3).

### Qué tienen en común las rígidas: pocos pasos distintos

Mirando sólo las soluciones con núcleo (N=7..11):

- **Son progresiones aritméticas a trozos.** P. ej. N=10 `[1,3,5,7,9,0,2,4,6,8]` (paso +2 y
  un salto), N=11 `[1,4,7,10,8,5,2,0,3,6,9]` (+3, −3, +3), N=11
  `[2,4,6,8,10,1,3,0,9,7,5]` (+2 y luego −2), N=9 `[1,4,7,5,8,2,0,3,6]` (pasos +3 y −2).
- **Pasos distintos** D = nº de valores de σ(i+1) − σ(i) mod N (mínimo sobre las 8
  simetrías). Media por grupo: toroidales 1 · rígidas 2–3 · núcleo parcial 3–4 · sin
  núcleo 4–5 · no medidas (5–6 ondas) 5,4.
- **Simétricas** (clase de tamaño < 8): 21–100 % de las rígidas frente a ~1 % del resto.
- En las de 2 ondas una onda domina (amplitud 1,0 frente a 0,1–0,4): una "casi toroidal"
  más una corrección. En las rígidas de 3 ondas de N=11 suele haber dos ondas en la misma
  recta modular exacta (una es armónico de la otra).

**Relación general (todas las soluciones, no sólo las rígidas): ondas mínimas ≈ pasos
distintos.**

| N | media de D para K = 1, 2, 3, 4, 5, 6 | correlación | K = D | \|K − D\| ≤ 1 |
|---|---|---|---|---|
| 8 | –, 2,0, –, 4,0 | 0,69 | 65 % (azar 56 %) | 100 % (89 %) |
| 9 | –, 2,0, 3,1, 3,7, 5,2 | 0,81 | 73 % (44 %) | 100 % (81 %) |
| 10 | –, 2,0, 2,0, 4,0, 5,2 | 0,77 | 55 % (34 %) | 100 % (81 %) |
| 11 | 1,0, 3,0, 3,0, 4,1, 5,4, 5,9 | 0,79 | 46 % (29 %) | 95 % (69 %) |

("azar" = barajando K entre clases; las dos magnitudes están concentradas, así que el
nivel base es alto, pero la relación lo supera con claridad.) Conecta con
`wave_queens/rotation.py`: las sturmianas eran justo "dos pasos adyacentes" ↔ 2 ondas.

---

## Camino inverso: filtrar por pasos y sólo entonces buscar ondas (`sospechosas.py`)

**Validación en N=11** (se conocen todos los mínimos): "sospechosa" = clase no toroidal con
≤ 3 pasos distintos → 35 clases (10 %), de las que 27 tienen ≤ 3 ondas (precisión 77 %);
recupera 27 de las 29 clases de ≤ 3 ondas y 12 de las 14 rígidas. Con ≤ 4 pasos: 83
clases, recupera 29/29 y 14/14 (precisión 35 %).

**Aplicado a N sin datos previos** (MILP sólo sobre las sospechosas, 120 s):

| N | clases | sospechosas (2 ≤ D ≤ 3) | 2 ondas | 3 ondas | 4 ondas | 5 ondas | tiempo |
|---|---|---|---|---|---|---|---|
| 12 | 1787 | 15 (0,8 %) | 3 | 4 | 8 | — | 1 min, todo exacto |
| 13 | 9233 | 108 (1,2 %) | **16** | 32 (1 cota) | 48 (cota) | 11 (cota) | ~15 min |

- Primeros mínimos exactos de N=12 y N=13 (antes: N=13 a ciegas no cerraba ninguna
  clase en 5 min). N=13: 124 soluciones de 2 ondas y 252 de 3, no toroidales.
- Se mantiene K ∈ {D−1, D, D+1}: D=2 → 2–3 ondas; D=3 → 2–5.
- Ejemplos de 2 ondas: N=12 `[1,3,5,7,9,11,0,2,4,6,8,10]`; N=13
  `[4,8,10,12,1,3,5,7,9,11,0,2,6]` (paso +2 con los extremos desplazados).

---

## Reglas: soluciones por tramos, construidas sin enumerar (`tramos.py`, `familias.py`)

Tramo = filas consecutivas con el mismo paso σ(i+1) − σ(i) (mod N). `tramos.py` construye
todas las soluciones con ≤ R tramos por backtracking (N=34 en 12 s con R=3).

**Mínimo de tramos según N mod 6** (N = 4..34, R ≤ 3; R ≤ 5 hasta N=21):

| N mod 6 | mínimo de tramos | cuántas |
|---|---|---|
| 1, 5 | **1** (toroidales lineales) | muchas |
| 2, 4 | **3** | crece con N |
| 0 | **3** | exactamente 2 (la clásica `1,3,5,…,0,2,4,…` y su espejo) |
| 3 | **5** (ninguna con ≤ 4) | N=9: 104, N=15: 28 |

De exactamente 2 tramos no hay ninguna en ningún N. Los mínimos son siempre impares.

**Familia cerrada (N ≡ 2, 4 mod 6): dos mitades de paso +2 unidas por un salto j**
(longitudes N/2−1, 1, N/2−1). Comprobado hasta N=1000 (`familias.py`):

- `j = 5 + 6m` (m = 0, 1, 2, …) da **exactamente N − 7 − 18m** posiciones iniciales;
- `j = 1` da exactamente 1 solución cuando N ≡ 0 o 4 (mod 6).

**Y sus ondas también tienen fórmula** (j = 5): `k₁ = (1, N/2)` y `k₂ = (N − 2v, v)` con v =
impar más cercano a N/3 (k₂ cumple u + 2v = N: es un armónico de la onda natural del
paso 2). Levita las N − 7 soluciones en todos los N probados (8…100) **sin búsqueda**.
El margen cae rápido (0,09 en N=10, 0,002 en N=40, 0,0001 en N=100, aprox. ∝ N⁻³): 2
ondas bastan matemáticamente para todo N, pero cada vez con menos holgura.

Otras familias de 3 tramos vistas: (s, j, s) con s = 4, 6, 8 en N concretos (16, 22, 26);
en N ≡ 1, 5: "sube y baja" (s, j, −s) con longitudes desiguales, y (s, j, s) con
longitudes (1, N−3, 1).

### Catálogo de 3 tramos por patrón de alfas, N = 4..46 (`catalogo.py`)

Alfa = paso de un tramo. Forma: L = tramo largo (≥ 2 pasos), j = salto (1 paso). Firmas
canónicas (se identifican espejo y reverso; los recuentos incluyen esas variantes).

| familia | forma | alfas | N | nº de soluciones | comprobado |
|---|---|---|---|---|---|
| A | L j L | (2, 5+6m, 2), mitades iguales | N ≡ 2, 4 (mod 6) | 2·(N − 7 − 18m) | N ≤ 1000 |
| B | L j L | (2, 1, 2) | N ≡ 0, 4 (mod 6) | 2 | N ≤ 1000 |
| C | j L j | (4, 2, 4), longitudes (1, N−3, 1) | N ≡ 1 (mod 6), N ≥ 13 | **2(N + 2)/3** | 13…43 |
| D | L j L "sube y baja" | ((N−1)/2, 2, −(N−1)/2), long. (3, 1, N−5) | N ≡ 5 (mod 6), N ≥ 11 | **4** | 11…41 |
| E | L j L "sube y baja" | ((N+1)/3, (N−5)/3, −(N+1)/3), long. (4, 1, N−6) | N ≡ 5 (mod 6), N ≥ 11 | **4** | 11…41 |
| F | j L j | (3, −(N−3)/2, 3) | irregular: 7, 17, 19, 29, 31, 37, 41 (no 11, 13, 23, 25, 35, 43) | 2, 6, 4, 14, 16, 34, 26 | sin regla aún |

- N ≡ 3 (mod 6): ninguna de ≤ 3 tramos (necesitan 5). N ≡ 0: sólo la B.
- Con 3 tramos hay soluciones de 2 alfas distintos (A, B, C, F) y de 3 (D, E: s, j, −s).
- Quedan "otras" sin clasificar que crecen con N (p. ej. 122 en N=41): alfas 4, 6, 8…
  en pares con salto que depende de N.

### Paso 1: las tres formas de 3 tramos y sus leyes (N ≤ 46 clasificado al 98,5 %)

| forma | N | alfas | ley |
|---|---|---|---|
| **G — dos mitades** (L j L) | par | (s, j, s), s par, mitades iguales | salto `j = s²+s−1 + 2m(s²−1)` → **`N − (s²+s+1) − 2m(s+1)²`** soluciones (×2 con el espejo), si `mcd(N, s(s−1)(s+1)) = 2`; si no, ninguna. Además j = 1 da 1 en ciertos N. |
| **H — extremos desplazados** (j L j) | impar, no múltiplo de 3 | (2b, b, 2b), longitudes (1, N−3, 1) | b = 2, N ≡ 1 (mod 6): **(N+2)/3** por signo. Otros b: existen pero irregulares (p. ej. b = 4: N = 23, 41, 53 con (N−20)/3; no en 29, 47, 59). |
| **V — sube y baja** (L j L) | N ≡ 5 (mod 6) | ((N−1)/2, 2, −(N−1)/2) y ((N+1)/3, (N−5)/3, −(N+1)/3) | 4 + 4 |

- La ley de G se comprobó con s = 2…12 y todo N par ≤ 260 (m = 0), y con m ≥ 1 en N ≈ 200–300:
  s=2: saltos 5, 11, 17… cuentas N−7, −18 cada vez; s=4: 19, 49, 79… N−21, −50; s=6: 41,
  111… N−43, −98; s=8: 71, 197… N−73, −162.
- La antigua familia A es G con s = 2; C es H con b = 2; F es H con b = −(N−3)/2.
- Reparto N ≤ 46: G 1194, H 514, V 76, sin clasificar 28 (alfas (s, j, s) con longitudes
  desiguales tipo (2, N−5, 2), (5, 6, 5), (11, 8, 11)).
- **La paridad sí importa aquí** (G sólo par, H y V sólo impar) y **la divisibilidad**
  (condición de mcd en G; N mod 6 en H y V). Primo vs compuesto no añade nada visible.

### Paso 2: 4–5 tramos (N = 4..22) y el principio de los bloques

`python tramos.py --R 5 --desde 4 --hasta 22` (N=22 en 100 s).

- **Más alfas**: con 5 tramos hay soluciones de 2, 3, 4 y hasta 5 alfas distintos (N=7: 4,
  N=13: 12, N=19: 12 con los 5 distintos). Con 4 tramos, 3–4 alfas.
- **N ≡ 3 (mod 6)** (9, 15, 21): no hay nada con ≤ 4 tramos; con 5 hay 104, 28, 76.
- N ≡ 0 (mod 6): con 5 tramos N=18 pasa de 2 a 76; N=12 sólo suma 2.

**El principio que unifica G y los múltiplos de 3: bloques.** Un tramo de paso s recorre
N/mcd(s, N) columnas antes de repetir, así que una solución "de paso s" se parte en
**g = mcd(s, N) bloques iguales unidos por g − 1 saltos → 2g − 1 tramos**:

| g | forma | tramos | N |
|---|---|---|---|
| 1 | un solo tramo (toroidal) | 1 | mcd(N, 6) = 1 |
| 2 | dos mitades de paso s par (familia G) | 3 | N par |
| 3 | **tres tercios de paso 3** | 5 | N múltiplo de 3 |

Por eso el mínimo de tramos es 1, 3 o 5 (siempre impar) según N mod 6.

**Leyes de los tres tercios de paso 3** (longitudes N/3−1, 1, N/3−1, 1, N/3−1), N ≡ 3 (mod 6),
comprobadas hasta N = 300 sin fallos:

| saltos (j₁, j₂) | soluciones |
|---|---|
| (7, 7) | **N − 13** |
| (11, 11) | **N − 29** |
| (7, 19) y (19, 7) | **N − 37** cada una |
| (2, 2) | 1–2, esporádica |

En N ≡ 0 (mod 6) los tercios son irregulares: N ≡ 0 (mod 12) sólo la (2, 2); N = 18, 42,
54 tienen muchas (18, 119, 135), N = 30 sólo 10.

### Nivel 5 y 7: cinco quintos de paso 5, siete séptimos de paso 7 (`bloques.py`)

**Predicción fallida** (anotada antes de calcular): extrapolando g = 2 (salto 5, N−7) y g = 3
(salto 7, N−13) a "salto 2s+1, N−(s²+s+1)" se esperaba salto 11 y N−31 para g = 5. No
ocurre: (11, 11, 11, 11) sólo da 4 soluciones en N = 25 y 35 y ninguna en 55, 65.

**Lo que sí hay** — ley exacta, de otra naturaleza:

> N = g·p con mcd(g, p) = 1 (g = 5 o 7): g bloques de paso g con todos los saltos iguales a
> **j = g + m·p**, para **m = 2 … g−2**, dan **exactamente N soluciones cada uno** (cualquier
> columna inicial), todas **toroidales no lineales**. Con m = 1 o g−1, ninguna.

Comprobado: g = 5 con p = 7, 11, 13, 17, 19, 23, 29, 49 (2N soluciones); g = 7 con p = 5, 11,
13, 17, 19, 23, 25, 29 (4N). Si p comparte factor con g (25, 125, 175; 49, 245) no hay.

Lectura: como mcd(g, p) = 1, Z_N ≅ Z_g × Z_p y estas soluciones son el **producto de dos
toroidales** (la de paso g sobre p filas y la lineal de multiplicador m sobre los g
bloques): la misma construcción σ⊗τ de `fractal_queens`, vista como bloques.

**Dos regímenes de bloques**:

| g | toroidal posible | tipo de ley | nº de soluciones |
|---|---|---|---|
| 2, 3 | no (factores de 6) | "defecto": saltos fijos | N − constante |
| 5, 7 | sí | producto de toroidales | N por cada m = 2…g−2 |

Sin explicar: los quintos con saltos desiguales en N par (N = 20: 28, 40: 132, 50: 64, 70:
317, 80: 736) y N = 25 (16).

### Estadísticas de alfas y tramos sobre TODAS las soluciones, N = 8..13 (`estadisticas_alfas.py`)

alfa = σ(i+1) − σ(i) mod N con signo en (−N/2, N/2]; tramo = racha de alfas iguales.

| N | soluciones | tramos (media / máx. posible) | con ≤ 5 tramos | tramos de longitud 1 | pasos en tramos largos | alfas distintos (media) |
|---|---|---|---|---|---|---|
| 8 | 92 | 5,7 / 7 | 34 (37 %) | 86 % | 29 % | 4,3 |
| 9 | 352 | 6,0 / 8 | 104 (30 %) | 67 % | 50 % | 3,9 |
| 10 | 724 | 7,6 / 9 | 58 (8 %) | 87 % | 26 % | 4,7 |
| 11 | 2680 | 8,2 / 10 | 160 (6 %) | 86 % | 30 % | 5,5 |
| 12 | 14200 | 9,8 / 11 | 4 (0,03 %) | 89 % | 21 % | 6,3 |
| 13 | 73712 | 10,5 / 12 | 308 (0,4 %) | 89 % | 22 % | 6,8 |

- **La mayoría de soluciones casi no tiene tramos**: cambia de alfa en casi cada fila. Las de
  pocos tramos son una cola que se adelgaza rápido con N.
- **Preferencia de alfas, suave**: |alfa| = 1 casi nunca (0,1× el azar; sólo "dando la vuelta");
  2 y 3 algo favorecidos (1,2–1,6×); el resto ≈ 1×.
- N=9 es distinto (50 % de los pasos en tramos largos, un tercio de los tramos de longitud 2):
  la huella de los tercios.
- **Tramos ↔ ondas**: correlación floja en N = 8–10 (0,1–0,3), 0,69 en N=11. Predice mejor el
  número de **alfas distintos** (0,7–0,8) que el de tramos: una solución puede alternar muchos
  tramos cortos usando pocos alfas y seguir siendo de pocas ondas.

### ¿Cuántas soluciones dan las reglas? (`reglas.py`, N = 4..60)

Soluciones generadas por fórmula (validadas una a una), distintas, y tras aplicar las 8
simetrías del tablero. T = toroidal de un tramo, G = dos mitades, H = extremos desplazados
(b=2), V = sube y baja, 3 = tres tercios, P = productos de 5/7 bloques.

| N | T | G | H | V | 3 | P | con simetrías | Q(N) | fracción |
|---|---|---|---|---|---|---|---|---|---|
| 4 | · | 2 | · | · | · | · | 2 | 2 | 100 % |
| 5 | 10 | · | · | · | · | · | 10 | 10 | 100 % |
| 6 | · | 2 | · | · | 2 | · | 4 | 4 | 100 % |
| 7 | 28 | · | 2 | · | · | · | 32 | 40 | 80 % |
| 8 | · | 2 | · | · | · | · | 4 | 92 | 4,3 % |
| 9 | · | · | · | · | 8 | · | 16 | 352 | 4,5 % |
| 10 | · | 8 | · | · | · | · | 16 | 724 | 2,2 % |
| 11 | 88 | · | · | 8 | · | · | 104 | 2680 | 3,9 % |
| 12 | · | 2 | · | · | 2 | · | 8 | 14200 | 0,06 % |
| 13 | 130 | · | 10 | · | · | · | 150 | 73712 | 0,2 % |
| 20 | · | 26 | · | · | · | · | 52 | 3,9·10¹⁰ | 10⁻⁹ |
| 25 | 250 | · | 18 | · | · | · | 286 | 2,2·10¹⁵ | 10⁻¹³ |
| 35 | 280 | · | · | 8 | · | 420 | 1136 | | |
| 47 | 2068 | · | · | 8 | · | · | 2084 | | |
| 58 | · | 304 | · | · | · | · | 608 | | |
| 59 | 3304 | · | · | 8 | · | · | 3320 | | |
| 60 | · | 2 | · | · | 2 | · | 8 | | |

- **Todo N de 4 a 60 recibe al menos una solución construida por regla** (los múltiplos de 6
  son los más pobres: siempre 8).
- Por clase: N ≡ 1, 5 (mod 6) cientos–miles (dominan las toroidales, ~N²/algo); N ≡ 2, 4
  decenas–cientos (G, crece con N); N ≡ 3 decenas–cientos (tercios); N ≡ 0 sólo 8.
- Como fracción de Q(N): 100 % hasta N=6, 80 % en N=7, ~2–4 % en N = 8–11, y después
  despreciable (Q(N) crece como N!, las reglas como N²).

### N primo: pendientes por clases (residuos cuadráticos) — `primos_residuos.py`

Para N primo las columnas forman un cuerpo y hay fórmulas más ricas que `a·i + c`.

**`σ(i) = a·i^k + b`**: el único exponente que da soluciones es **k = (p+1)/2**, y sólo si
**p ≡ 1 (mod 4)** (13, 17, 29, 37, 41), con exactamente **(p−5)/2** valores de a. Como
`i^((p+1)/2) = ±i` según i sea o no residuo cuadrático, es un "sube y baja" de pendientes
+a / −a gobernado por los cuadrados módulo p. (`a/(i+c) + b` no da ninguna.)

**General — dos pendientes**: `σ(i) = a·i` si i es residuo cuadrático, `b·i` si no, más todas
las traslaciones `σ(i+c) + b`. Todas toroidales, ninguna de un solo tramo:

| p | lineales p(p−3) | pares (a, b) | soluciones (pares·p²) | total 1 y 2 pendientes |
|---|---|---|---|---|
| 7, 11 | 28, 88 | 0 | 0 | 28, 88 |
| 13 | 130 | 8 | 1352 | **1482** |
| 17 | 238 | 14 | 4046 | 4284 |
| 19 | 304 | 16 | 5776 | 6080 |
| 23 | 460 | 32 | 16928 | 17388 |
| 29 | 754 | 72 | 60552 | 61306 |
| 31 | 868 | 72 | 69192 | 70060 |
| 37 | 1258 | 112 | 153328 | 154586 |
| 41 | 1558 | 158 | 265598 | 267156 |
| 43 | 1720 | 160 | 295840 | 297560 |

- N=13: de las 4524 toroidales, 1482 son de 1–2 pendientes; añadiendo 3 pendientes (clases
  cúbicas) se llega a 2496 (55 %). Las otras 2028 necesitan más clases.
- Tres pendientes distintas (clases cúbicas, p ≡ 1 mod 3): 0 en 7, 13, 19; 156 en 31, 312 en
  37, 528 en 43.
- Sobre Q(13) = 73712: las reglas pasan de 150 (0,2 %) a ~2500 (3,4 %).

### Primos: la condición exacta y la escalera de clases (`clases_primos.py`)

`σ(i) = a_{clase(i)}·i (mod p)`, con `clase(i) = (log discreto de i) mod d`, d | p−1.

**Condición exacta** (sin fuerza bruta). Sea L(x) = clase(x). La solución es toroidal ⇔ las
tres aplicaciones `j → j + L(a_j)`, `j → j + L(a_j + 1)`, `j → j + L(a_j − 1)` son
permutaciones de Z_d. Elegir las pendientes es un problema de reinas **en Z_d**, cuyas piezas
son los valores de a clasificados por su **tipo** `(L(a), L(a+1), L(a−1))`.

- **d = 2**: a y b valen ⇔ son del mismo tipo (residuo o no) en a, a+1 y a−1 a la vez. Nº de
  pares = **Σ n_t(n_t − 1)** sobre los 8 tipos. Reproduce 8, 14, 16, 32, 72, 72, 112, 158,
  160, 200, 288 para p = 13…53 (p=43: 5 de cada tipo → 8·5·4 = 160). Como n_t ≈ (p−3)/8,
  crece como p²/8.
- **d = p − 1** (una clase por fila) cuenta **todas** las toroidales por (0,0): 4, 8, 348 para
  p = 7, 11, 13 → ×p = 28, 88, 4524, los valores conocidos.
- Generación y conteo verificados contra fuerza bruta (p = 13, 17, 19, 31; d = 2, 3).

**Cobertura de las toroidales por escalón** (con traslaciones):

| p | toroidales | d=1 | hasta d=2 | hasta d=3 | hasta d=4 | hasta d=6 |
|---|---|---|---|---|---|---|
| 13 | 4524 | 2,9 % | 32,8 % | 55,2 % | 55,2 % | **100 %** |
| 17 | 140692 | 0,2 % | 3,0 % | — | 6,3 % | — (d=8: 180 por el origen) |
| 19 | 820496 | 0,04 % | 0,7 % | 1,8 % | — | 1,8 % (d=9: 544) |

En N=13 **todas** las toroidales son "≤ 6 pendientes por clase + traslación". En 17 y 19 la
escalera baja cubre poco: el número de toroidales explota y hacen falta los escalones altos.

Cobertura completa por escalón (todas las clases intermedias generadas):

| p | filas no nulas | pendientes: cobertura de las toroidales |
|---|---|---|
| 13 | 12 | 1: 2,9 % · 2: 32,8 % · 3: 55,2 % · 4: 55,2 % · **6: 100 %** |
| 17 | 16 | 1: 0,2 % · 2: 3,0 % · 4: 6,3 % · **8: 34,3 %** · (16: 100 %, trivial) |
| 19 | 18 | 1: 0,04 % · 2: 0,7 % · 3: 1,8 % · 6: 1,8 % · **9: 24,0 %** · (18: 100 %, trivial) |

"Trivial" = una pendiente por fila: cualquier permutación se escribe así, no es una regla.
En N=13 basta la mitad de pendientes que de filas para tenerlas todas; en 17 y 19 la mitad
da sólo el 34 % y el 24 %.

---

## Primos: regla unificada "un tramo + filas reflejadas" (`reflejos.py`)

```
σ(i) = a·i + c            si i ∉ R
σ(i) = a·(m − i) + c      si i ∈ R        (mod p),   R simétrico respecto a m
```

Reflejar una fila es darle la pendiente −a. Unifica lo que antes eran reglas sueltas:

| caso | R | centro m |
|---|---|---|
| extremos desplazados (H) | {0, p−1} | p−1 |
| sube y baja (V) | bloque del borde: {0,1,2,3} o {0,1,3,4} | 3 o 4 |
| toroidal de pendientes +a/−a (p ≡ 1 mod 4) | no residuos cuadráticos | 0 |
| toroidal de un tramo | vacío, o todas las filas | — |

Todos los R simétricos, por fuerza bruta:

| p | soluciones | toroidales | no toroidales | filas reflejadas → soluciones |
|---|---|---|---|---|
| 7 | 32 de 40 | 28 (todas) | 4 | 2: 4 |
| 11 | 192 | 88 (todas) | 104 | 2: 20 · 4: 84 |
| 13 | 2072 | 1820 de 4524 | 252 | 2: 108 · 4: 1142 · 6: 692 |

**Criterio exacto para el caso toroidal (sin validar).** Cada fila reflejada i tiene dos
"compañeras de diagonal": `k⁺(i) = ((1−a)·i + a·m)/(1+a)` y `k⁻(i) = ((1+a)·i − a·m)/(1−a)`.
Entre filas reflejadas nunca hay conflicto; con una no reflejada sólo si es su compañera.
⇒ **la solución es toroidal ⇔ R es unión de órbitas de {i → m−i, k⁺, k⁻}**. Coincide con
la fuerza bruta en p = 7, 11, 13 y da: 28, 88, 1820, 5440, 4636, 460 para p = 7…23 (en 23
sólo las de un tramo: las órbitas dependen del orden de λ = (1−a)/(1+a) mod p).

**Caso no toroidal**: R no contiene alguna compañera, y entonces la solución vale sólo para
los desplazamientos c en que esa coincidencia de diagonal "da la vuelta" al tablero (un
intervalo cíclico de c por cada compañera). Con un solo par de filas (intercambio):
4, 20, 108, 324, 392, 904, 1584, 3736 soluciones para p = 7…31.

### Regla exacta general para los reflejos (toroidales o no), sin validar

Fijados pendiente a, centro m y desplazamiento c: cada fila reflejada i coincide en diagonal
(módulo p) con una fila k del tramo original por cada diagonal. El ataque es **real** sólo
si la coincidencia se da también como enteros; si "da la vuelta" al tablero no hay ataque.
Eso se decide con una comparación y define **obligaciones**: "si reflejas la unidad
{i, m−i}, debes reflejar también la de k" (o no puedes reflejarla, si k es la fila fija).

> Los conjuntos R válidos son exactamente los **cerrados bajo esas obligaciones**.

`all_by_rule(p)` los enumera sin callejones sin salida y construye cada solución sin
comprobarla. Coincide con la fuerza bruta en p = 7, 11, 13 y todas las generadas son válidas.

| p | total | toroidales | **no toroidales** | de un tramo | tiempo |
|---|---|---|---|---|---|
| 7 | 32 | 28 | 4 | 28 | 0 s |
| 11 | 192 | 88 | 104 | 88 | 0 s |
| 13 | 2072 | 1820 | 252 | 130 | 0 s |
| 17 | 7128 | 5440 | 1688 | 238 | 0 s |
| 19 | 7584 | 4636 | 2948 | 304 | 0 s |
| 23 | 10880 | 460 | 10420 | 460 | 1 s |
| 29 | 147960 | 116812 | 31148 | 754 | 3 s |
| 31 | 139336 | 81592 | 57744 | 868 | 4 s |
| 37 | 1088840 | 902060 | 186780 | 1258 | 23 s |
| 41 | 2252240 | 1927984 | 324256 | 1558 | 60 s |
| 43 | 1055248 | 534232 | 521016 | 1720 | 30 s |
| 47 | 846928 | 2068 | 844860 | 2068 | 28 s |

En p = 23 y 47 no hay toroidales por reflejo más allá de las de un tramo, pero sí muchas no
toroidales. El tiempo es proporcional al número de soluciones (no hay búsqueda).

---

## Foco: no toroidales con estructura de alfas, N primo (`pocos_alfas.py`)

**Cuánto cubren ya los reflejos** (clases no toroidales, por nº de alfas distintos):

| N | 2 alfas: clases / cubiertas | 3 alfas: clases / cubiertas | 4 alfas |
|---|---|---|---|
| 7 | 1 / 1 | — | 1 / 0 |
| 11 | 2 / 0 | 33 / 13 | 48 / 1 |
| 13 | 8 / 3 | 100 / 18 | 310 / 10 |

**Construcción directa de las de ≤ 2 alfas** (backtracking fila a fila restringido a 2 valores
de paso; no enumera todas las soluciones):

| p | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 |
|---|---|---|---|---|---|---|---|---|---|
| no toroidales de exactamente 2 alfas | 2 | 6 | 28 | 108 | 76 | 110 | 230 | 288 | 406 |
| palíndromos | 2 | 6 | 28 | 108 | 76 | 110 | 230 | 288 | 406 |

- **Todas son palíndromos** (la palabra de alfas se lee igual al revés ⇔ la solución es
  simétrica respecto al centro, módulo p). Ninguna de exactamente 2 alfas es toroidal.
- Forma de la palabra (alfa dominante A, alfa raro b), total p = 7…37 (1254 soluciones):
  escalera regular (los b equiespaciados, p. ej. `AbAAbAAbAAbA`) 474 · dos espaciados, tipo
  sturmiano (`bAAAbbAAAb`) 350 · sólo los dos extremos (`bAAAAAAAAAAb`, la antigua familia
  H) 270 · bloques iguales equiespaciados (`AbbAAbbAAbbAAbbA`) 124 · otras 36.

### Regla 1 de las no toroidales de 2 alfas: la escalera simétrica (`escaleras.py`)

Alfa dominante A, alfa raro b = A + δ. Escalones separados q pasos y colocados
simétricamente. **Ley exacta (sin excepciones en los datos): `q·A ≡ −2δ (mod p)`**, es decir,
el escalón queda forzado: `δ = −q·A/2`. Parámetros libres: A, q, nº de escalones k ≥ 2,
columna inicial c.

- k = 2: dos escalones simétricos (con el primero en el borde: los "extremos desplazados").
- k ≥ 3: escalera regular (`AbAAbAAbAAbA`).

Generada por la fórmula (cada candidata se valida; ~p³ candidatas, segundos):

| p | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 | 47 | 53 | 59 | 61 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| familia | 2 | 0 | 24 | 66 | 32 | 76 | 146 | 158 | 246 | 440 | 388 | 666 | 790 | 1158 | 922 |
| no toroidales de 2 alfas (catálogo) | 2 | 6 | 28 | 108 | 76 | 110 | 230 | 288 | 406 | | | | | | |

La familia está contenida en el catálogo y cubre el 60 % (750 de 1254 para p ≤ 37).

**Lo que queda de 2 alfas** (504): palabras con dos espaciados alternos (`bAAAbbAAAb`,
`AbbAAbbAAbbAAbbA`). No siguen una ley única; las dos subfamilias mayores cumplen
`(g₁+g₂)·A ≡ −δ` y `(g₁+g₂)·A ≡ −3δ` (170 soluciones cada una); el resto está repartido.

### El retraso: la lectura que unifica las de 2 alfas

Una solución de 2 alfas es una toroidal de pendiente A "releída con retraso":
`σ(i) = A·(i − h·n(i)) + c`, con n(i) = nº de escalones antes de la fila i y
**h = −δ/A (mod p)** el retraso por escalón. Medido como fracción del periodo P del patrón de
escalones:

| patrón | retraso h/P | soluciones (p ≤ 37) |
|---|---|---|
| escalera regular o dos escalones (1 escalón por periodo) | **1/2** siempre | 744 |
| dos espaciados alternos (2 escalones por periodo) | **1** | 170 |
| | **1/3** | 170 |
| | −2/5, −1/3, −5/2, −1, 1/5, ±4, … | 134 |

- Con h = P/2 la ley es `q·A ≡ −2δ`; además **q es siempre impar** (si fuera par, dos filas a
  distancia q/2 separadas por un escalón compartirían columna).
- No hay fórmula cerrada para qué A valen en cada p ni para los recuentos: la ley fija el
  escalón y el resto se comprueba (segundos).

**Cobertura de las no toroidales de 2 alfas** con las tres leyes (retraso 1/2, 1 y 1/3),
generadas por fórmula y validadas; nada generado cae fuera del catálogo:

| p | 7 | 11 | 13 | 17 | 19 | 23 | 29 | 31 | 37 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| catálogo | 2 | 6 | 28 | 108 | 76 | 110 | 230 | 288 | 406 | 1254 |
| cubierto | 2 | 6 | 28 | 104 | 66 | 108 | 194 | 234 | 340 | **1082 (86 %)** |

### Nivel de 3 alfas, N primo (`pocos_alfas.py --D 3`, `desplazamientos.py`)

| p | 7 | 11 | 13 | 17 | 19 | 23 |
|---|---|---|---|---|---|---|
| no toroidales de exactamente 3 alfas | 2 | 146 | 486 | 1682 | 2336 | 5490 |
| palíndromos | 2 | 30 | 98 | 386 | 680 | 1154 |
| toroidales de exactamente 3 alfas | 0 | 0 | 0 | 0 | 0 | 0 |

- Tampoco hay toroidales de exactamente 2 ni 3 alfas: las toroidales tienen 1 alfa o muchos.
- **Relación entre los dos escalones** δ₁, δ₂ (respecto al alfa dominante): en el **35 %**
  son opuestos, δ₂ = −δ₁ (alfas A, A+δ, A−δ; por azar sería ~5 %). El resto son cocientes
  pequeños (−2, −3/2, −3, 1/2, 2…), poco significativos con p ≤ 23.

**Regla "intercambios a distancia fija"** (lectura con retraso de los escalones opuestos):
un tramo `A·i + c` donde cada fila i de un conjunto U se intercambia con la fila i + d
(misma d para todas). Un solo par coincide con un reflejo. Construida fila a fila con poda:

| p | soluciones (todas no toroidales) | por nº de pares intercambiados |
|---|---|---|
| 13 | 224 | 1: 108 · 2: 12 · 3: 16 · 4: 38 · 5: 16 · 6: 34 |
| 17 | 836 | 1: 324 · 2: 248 · 3: 114 · … |
| 19 | 816 | 1: 392 · 2: 228 · 3: 48 · 4: 132 · … |
| 23 | 2056 | 1: 904 · 2: 520 · 3: 152 · 4: 144 · … |

**Cobertura del nivel de 3 alfas** (reflejos ∪ intercambios a distancia):

| p | 3 alfas | reflejos | intercambios | unión | de las de escalones opuestos |
|---|---|---|---|---|---|
| 11 | 146 | 60 | 20 | 60 (41 %) | 20 de 24 (83 %) |
| 13 | 486 | 124 | 180 | 206 (42 %) | 90 de 116 (78 %) |
| 17 | 1682 | 392 | 690 | 764 (45 %) | 534 de 653 (82 %) |
| 19 | 2336 | 418 | 668 | 722 (31 %) | 570 de 720 (79 %) |
| 23 | 5490 | 950 | 1852 | 1900 (35 %) | 1529 de 2040 (75 %) |

Sin regla todavía: ~20 % de las de escalones opuestos y casi todas las de escalones no
opuestos (δ₂ = −2δ₁, 2δ₁, …), que son el 60–65 % del nivel.

### Rotaciones de 3 filas (`desplazamientos.build3`)

Ternas (i, i+d, i+2d) que rotan sus columnas (escalones +δ, +δ, −2δ). **Aporta poco**:
cobertura del nivel de 3 alfas N=11 41 % → 56 %, pero en p = 13…23 sólo +0–1 %. No produce
ninguna de las soluciones "a 3 reinas de una toroidal" (8 en N=7, 24 en N=11, 3264 en N=13).
La hipótesis "el segundo escalón es una rotación simple" no se sostiene.

### Reglas ↔ ondas (N=11, mínimos de ondas de las 2680 soluciones)

Cada solución se asigna al primer grupo que la contiene (reglas cerradas bajo las 8
simetrías):

| grupo | soluciones | ondas mínimas | media | margen medio |
|---|---|---|---|---|
| toroidal de un tramo | 88 | 1: 88 | 1,00 | 0,088 |
| escaleras (2 alfas) | 12 | 3: 12 | 3,00 | 0,053 |
| reflejos (incluye los intercambios de un par) | 104 | 2: 20 · 3: 44 · 4: 40 | 3,19 | 0,045 |
| rotaciones de 3 | 136 | 3: 72 · 4: 32 · 5: 32 | 3,71 | 0,045 |
| **sin regla** | 2340 | 3: 60 · 4: 164 · 5: 1628 · 6: 488 | 5,09 | 0,035 |

**¿Cuántas de las de pocas ondas salen por regla?** 1 onda: 88/88 · 2 ondas: 20/20 ·
3 ondas: 128/188 (68 %) · 4 ondas: 72/236 (31 %) · 5 ondas: 32/1660 (2 %) · 6 ondas: 0/488.

Las 9 clases de reflejos con núcleo medido son todas **rígidas** (representación única).

**La onda principal es la de la toroidal de origen.** Según la distancia (en reinas) a la
toroidal de un tramo más cercana, de pendiente a (su onda cumple k₀ + a·k₁ ≡ 0):

| distancia | soluciones | onda principal = la de esa toroidal | alguna onda lo es | ondas (media) | 2ª onda / 1ª |
|---|---|---|---|---|---|
| 0 | 88 | 100 % | 100 % | 1,00 | — |
| 2 | 20 | 100 % | 100 % | 2,00 | 0,10 |
| 3 | 24 | 100 % | 100 % | 4,00 | 0,98 |
| 4 | 140 | 92 % | 100 % | 4,09 | 0,72 |
| 5 | 520 | 91 % | 100 % | 5,02 | 0,72 |
| 6 | 892 | 56 % | 98 % | 4,86 | 0,82 |
| ≥ 7 | 996 | 24 % | 82 % | 5,14 | 0,79 |

---

## Por qué no se llegaba al 100 % en 3 alfas, y el criterio que sí delimita: los CORTES (`cortes.py`)

Cobertura del nivel de 3 alfas con todas las reglas (cerradas bajo simetrías): 73 %, 49 %,
54 %, 34 %, 41 % para p = 11, 13, 17, 19, 23. Hipótesis descartadas para las pendientes:

- no están cerca de UNA toroidal de un tramo (les sobran 6–12 reinas) ni de DOS pegadas;
- sin relación dominante entre los dos escalones (−3/2, −3, −2, −1, 2, 1/2… a ~5–12 % cada una);
- no son "escalera + retoque": sólo 1–3 % tienen ambos tipos de escalón cumpliendo una ley
  de retraso;
- pero necesitan las mismas ondas que las explicadas (N=11: 3 ó 4), así que **sí** son del grupo.

**Lo que son**: una toroidal de un tramo **cortada en bloques de filas consecutivas y
recolocada**. Ej. N=11 `[1,3,6,8,10,5,0,2,4,7,9]` = 2τ+1 con τ = `[0,1 | 8,9,10 | 2 | 5,6,7 | 3,4]`.

**Cortes** = pasos que no son el alfa dominante = (bloques − 1). N=11 (mínimo de la clase):

| cortes | soluciones | con las reglas anteriores | ondas (media) |
|---|---|---|---|
| 0 | 88 | 100 % | 1,00 |
| 3 | 8 | 100 % | 2,00 |
| 4 | 188 | 64 % | 3,28 |
| 5 | 368 | 20 % | 4,46 |
| 6 | 932 | 4 % | 5,01 |
| 7 | 992 | 2 % | 5,29 |
| 8 | 104 | 0 % | 5,62 |

Correlación con las ondas mínimas: **0,83** (alfas distintos: 0,77; tramos: 0,69).

**Construcción completa** (`python cortes.py --J 4`): para cada A y columna inicial, fila a
fila se sigue con el paso A o se gasta un corte. Genera *todas* las soluciones con ≤ J cortes:

| N | ≤ 4 cortes | por nº de cortes | tiempo |
|---|---|---|---|
| 7 | 40 (= todas las de N=7) | 0: 28 · 2: 2 · 4: 10 | 0 s |
| 11 | 216 | 0: 88 · 3: 4 · 4: 124 | 0,1 s |
| 13 | 552 | 0: 130 · 2: 26 · 3: 56 · 4: 340 | 1 s |
| 17 | 1238 | 0: 238 · 2: 56 · 3: 160 · 4: 784 | 16 s |
| 19 | 1544 | 0: 304 · 2: 78 · 3: 104 · 4: 1058 | 57 s |
| 23 | 2956 | 0: 460 · 2: 132 · 3: 208 · 4: 2156 | 8 min |

Nunca aparece una solución con exactamente 1 corte. El coste crece como ~N^(J+2): es una
búsqueda con poda, no una fórmula.

---

## ¿Las soluciones complejas son combinación de simples? (`combinaciones.py`)

**Intersección de dos tableros de franjas paralelas.** Si la solución cabe en t₁ franjas de
pendiente a₁ y en t₂ de pendiente a₂, cabe en su intersección, de t₁·t₂ casillas (N primo).
N=11: toroidales (1, 11) → 11 exacta; las de 2 ondas (3, 9) → 27; la mayoría (5–6, 5–6) →
25–36 casillas para 11 reinas. No es económica: la solución típica ocupa la mitad de las
franjas en cualquier dirección.

**Mezcla de toroidales de pendientes distintas.** cobertura = mínimo nº de toroidales de un
tramo cuya unión contiene las N reinas (cada fila toma su reina de una de ellas). Exacto:

| N | 1 | 2 | 3 | 4 | 5 | media | permutaciones al azar (media) |
|---|---|---|---|---|---|---|---|
| 7 | 28 | 12 | | | | | |
| 11 | 88 (3 %) | 104 (4 %) | 912 (34 %) | 1576 (59 %) | — | 3,48 | 4,08 (4: 81 %, 5: 11 %) |
| 13 (muestra 20000) | 0,2 % | 3,7 % | 8,4 % | 66 % | 22 % | 4,06 | 4,67 (4: 34 %, 5: 64 %) |

- Toda solución de N=11 es mezcla de ≤ 4 toroidales, y de N=13 de ≤ 5; pero una permutación
  al azar también se cubre con 4–5. La ventaja de las soluciones es real y pequeña: ~0,6
  toroidales menos de media, y ~5,2 reinas en la toroidal más compartida frente a ~4,3–4,5.
- N=11: correlación cobertura–ondas 0,71; las de 2 toroidales necesitan 3,2 ondas de media,
  las de 3 → 4,7, las de 4 → 5,2. Reparto típico de reinas: (7, 5) y (9, 3) con 2; (6, 4, 4)
  con 3; (4, 4, 3, 2) con 4.

### Intersección y XOR de dos tableros de ondas (`dos_tableros.py`)

Tablero = casillas donde un campo de pocas ondas supera su umbral (cualquier nº de casillas
encendidas). MILP exacto: mínimo nº TOTAL de ondas (tablero 1 + tablero 2) tal que la
intersección (o el XOR: casillas encendidas en exactamente uno) sea la solución.
(El XOR de dos *soluciones* enteras no puede ser solución: cada fila donde difieren aportaría
dos reinas; por eso los tableros son patrones de franjas, no soluciones.)

| N | clases | directo (1 campo) | intersección (k₁+k₂) | XOR |
|---|---|---|---|---|
| 7 | 6 | 1, 1, 5, 1, 2, 1 | 1, 1, 5 (3+2), 1, 2, 1 | igual |
| 8 | 12 | 2 y 4 | 2 (1+1) y 4 (2+2 ó 3+1); una clase 5 (cota) | 4 (2+2) en 4 clases; resto sin cerrar en 150 s |

- **El total de ondas no baja**: la intersección necesita las mismas que el método directo
  (óptimo probado en 16 de 18 clases, con ≤ 3 ondas por tablero).
- **Sí reparte**: una solución de 4 ondas se obtiene como intersección de dos tableros de 2
  ondas cada uno (con 16–36 casillas encendidas cada uno). N=8 clase 9: dos tableros de
  **1 onda** (32 y 16 casillas) cuya intersección es la solución.
- XOR: donde cierra, tampoco baja de 4; tableros casi vacíos (3–5 casillas) en esos casos.

---

## Fusión de soluciones con una puerta (`puerta.py`, `fusion.py`)

Idea de Carl: una solución como mezcla de otras dos, eligiendo filas de una u otra con un
criterio. Criterio probado: una **onda-puerta** a lo largo de las filas (abre las filas cuya
fase u·i mod N cae en un arco). Piezas: soluciones cualesquiera, no sólo toroidales.

`T(i) = A(i)` en las filas abiertas, `B(i)` en las cerradas, con A ≠ T ≠ B soluciones.

| N | soluciones | son fusión de otras dos (puerta de onda) | con puertas al azar |
|---|---|---|---|
| 7 | 40 | 20 % | 25 % |
| 11 | 2680 | 67 % | 65 % |
| 13 | 73712 | **98,8 %** | 98,7 % |

- En N=13 casi toda solución se arma con ~6 filas de una y ~7 de otra. El tipo de puerta no
  importa (al azar da lo mismo): las soluciones comparten mitades entre sí.
- N=11 por ondas mínimas: 2 ondas 100 %, 3: 57 %, 4: 76 %, 5: 70 %, 6: 62 %.

**Cierre por fusión** (partir de una semilla y añadir toda solución que sea fusión de dos ya
obtenidas). Semilla: sólo las toroidales de un tramo.

| N | semilla | alcanzadas | rondas |
|---|---|---|---|
| 11 | 88 | 320 (12 %) — se estanca en la ronda 4 | 4 |
| 13 | 130 | **72532 de 73712 (98,4 %)** | 36 |

N=13: arranque lento (934 en la ronda 17) y luego reacción en cadena (3618 en la 25, 43756 en
la 30, 72524 en la 35). Sólo con puertas de bloque (frecuencia 1) se queda en 224 (0,3 %):
hacen falta las puertas periódicas. Quedan 1180 soluciones nunca alcanzadas.

Con piezas toroidales únicamente y puerta de una onda (`puerta.py`): 4, 104, 236 no
toroidales para N = 7, 11, 13 (N=11: las mismas 104 que dan los reflejos).

### Orden constructivo por niveles (`niveles.py`)

Nivel 0 = soluciones por reglas; nivel k+1 = fusiones (puerta de onda sobre las filas) de dos
soluciones de niveles ≤ k, más sus 8 simetrías.

**N=13, semilla = reglas** (toroidales de un tramo + escaleras + reflejos + intercambios +
rotaciones): **99,5 % en 7 niveles**.

| nivel | soluciones | acumulado | cortes (media) |
|---|---|---|---|
| 0 | 2384 | 2384 | 6,53 |
| 1 | 6876 | 9260 | 8,20 |
| 2 | 13556 | 22816 | 8,53 |
| 3 | 24140 | 46956 | 8,59 |
| 4 | 21464 | 68420 | 8,59 |
| 5 | 4532 | 72952 | 8,58 |
| 6 | 384 | 73336 | 8,52 |
| 7 | 32 | 73368 | 8,12 |
| nunca | 344 | | 8,63 |

(Con semilla sólo de un tramo: 98,4 % en 36 niveles; la semilla rica acorta de 36 a 7.)

**N=11**: se estanca. Semilla de un tramo: 320 (12 %); semilla de reglas (340): 752 (28 %) en
4 niveles. Nivel 0: 2,8 ondas de media; niveles 1–4: 5,0–5,2; nunca alcanzadas: 5,1.

- El orden existe (generaciones) y en N=13 es casi completo, pero **los niveles no ordenan
  por complejidad**: del nivel 1 en adelante todos tienen la misma (cortes ≈ 8,5; ondas ≈ 5
  en N=11). Sólo el nivel 0 es distinguible.
- Aviso: el cálculo usa la lista de soluciones para saber qué fusiones son válidas. Muestra
  que el orden existe; aún no es un generador a ciegas.

### Fusión a ciegas con índice exacto (`fusion_ciega.py`)

Sin lista de soluciones. T = A en las filas abiertas + B en las cerradas es solución ⇔
(1) las columnas de B en las cerradas son **exactamente** las que le faltan a A; (2) y (3)
las diagonales de las dos partes son disjuntas. (1) es una igualdad ⇒ índice por puerta
`{máscara de columnas en las filas cerradas → soluciones}`; dada A se va directo a su cajón
complementario y (2), (3) son dos AND de enteros.

**N=13** — reproduce exactamente los niveles de `niveles.py` (+6876, +13556, +24140):

| nivel | pares comprobados | frente a todos-con-todos | válidos |
|---|---|---|---|
| 1 | 1,7·10⁷ | 131× menos | 36 % |
| 2 | 2,0·10⁸ | 170× menos | 37 % |
| 3 | 1,0·10⁹ | 194× menos | 38 % |

El índice quita > 99 % de las comparaciones; lo que queda es **redundancia** (nivel 3: 4·10⁸
fusiones válidas para 24140 soluciones nuevas).

**N=17** (95,8 millones de soluciones, no enumeradas), usando 3000 piezas al azar por nivel:

| nivel | nuevas | total | tiempo |
|---|---|---|---|
| 0 (reglas + simetrías) | | 8620 | 1 s |
| 1 | +33974 | 42594 | 21 s |
| 2 | +40136 | 82730 | 20 s |
| 3 | +30362 | 113092 | 19 s |

Todas válidas; ~250–350× menos comparaciones que todos-con-todos y ~40 % de aciertos.

### ¿Bins para decidir qué combinar? (`bins.py`) y fusión por mitades

**Categorías probadas** (N=13, muestra de 4000, todas las fusiones válidas entre ellas, tasa
relativa a la media):

- bin = |alfa dominante|: todas las tasas entre 0,96 y 1,12 (mismo bin: 22 % de las fusiones,
  por azar 20 %).
- bin = cortes: 0,7–1,1 en todos los bins con tamaño apreciable (mismo bin 30 % vs 29 %).

⇒ **Ni la pendiente de fondo ni la complejidad predicen si dos soluciones se fusionan**: el
espacio está bien mezclado; no hay bins "inteligentes" de ese tipo.

**Lo que sí elimina alternativas: fusionar mitades, no soluciones** (`--mitades`). Muchas
soluciones comparten exactamente la misma mitad (las filas abiertas, o las cerradas). Guardando
por puerta las mitades *distintas*, cada solución resultante se obtiene una sola vez por
puerta. Mismos niveles, N=13:

| nivel | pares comprobados: soluciones | mitades | tiempo: soluciones | mitades |
|---|---|---|---|---|
| 1 | 1,7·10⁷ | 4,6·10⁶ | 6 s | 3 s |
| 2 | 2,0·10⁸ | 2,8·10⁷ | 67 s | 13 s |
| 3 | 1,0·10⁹ | 9,4·10⁷ | 388 s | 36 s |
| 4 | — | 2,6·10⁸ | — | 95 s |

Frente a todos-con-todos: 480×, 1190×, 2169×, 3292× menos comparaciones.

### Plantillas y compatibilidad media

Plantilla (puerta) = filas con fase u·i mod N en un arco [x, x+w): u = 1…N/2, x = 0…N−1,
w ≥ 2 a cada lado. No son al azar: se usan todas (N=11: 220, N=13: 390, N=17: 952).

Si de una solución A se **conservan k filas** (las que marca la plantilla) y se cambian las
demás, nº medio de soluciones distintas que se pueden formar (por solución y plantilla):

| k conservadas | N=11 | con alguna | N=13 | con alguna |
|---|---|---|---|---|
| 2 | 31,0 | 100 % | 573,3 | 100 % |
| 3 | 5,28 | 97 % | 72,6 | 100 % |
| 4 | 1,27 | 65 % | 12,4 | 99 % |
| 5 | 0,41 | 31 % | 3,01 | 85 % |
| 6 | 0,15 | 14 % | 0,96 | 52 % |
| 7 | 0,066 | 6 % | 0,37 | 28 % |
| 8 | 0,025 | 2,5 % | 0,16 | 14 % |
| 9 | 0,007 | 0,7 % | 0,072 | 7 % |

Cada fila conservada de más divide las opciones por ~3–8. Compañeras B por (solución,
plantilla), promediando todas las plantillas: N=11 9,6 de 2679 (0,36 %); N=13 134 de 73711
(0,18 %).

---

## Contar juntando mitades (`contar_mitades.py`) — exacto por construcción

Plantilla fija: filas de arriba (h = N//2) y de abajo. Cada solución se descompone de forma
única en sus dos mitades ⇒ Q(N) = nº de parejas (mitad de arriba, mitad de abajo) compatibles,
tomando como mitades **todas** las colocaciones sin ataques en esas filas (no se usa ninguna
solución conocida).

| N | filas | A (arriba) | B (abajo) | C (columnas complementarias) | **Q** | p_col = C/(A·B) | p_diag = Q/C |
|---|---|---|---|---|---|---|---|
| 8 | 4+4 | 344 | 344 | 1760 | 92 | 1,5·10⁻² | 0,052 |
| 9 | 4+5 | 732 | 1614 | 9216 | 352 | 7,8·10⁻³ | 0,038 |
| 10 | 5+5 | 3916 | 3916 | 58304 | 724 | 3,8·10⁻³ | 0,0124 |
| 11 | 5+6 | 8492 | 21362 | 365716 | 2680 | 2,0·10⁻³ | 0,0073 |
| 12 | 6+6 | 52856 | 52856 | 2772920 | 14200 | 9,9·10⁻⁴ | 0,0051 |
| 13 | 6+7 | 117694 | 335010 | 20897532 | 73712 | 5,3·10⁻⁴ | 0,0035 |
| 14 | 7+7 | 835056 | 835056 | 182527728 | 365596 | 2,6·10⁻⁴ | 0,0020 |

Q coincide con el valor conocido en todos los N = 4…14 (N=14 en 21 s).
`Q = A · B · p_col · p_diag`; p_col ≈ 1 / C(N, h) (N=14: 1/3432 = 2,9·10⁻⁴).

## Fusión usando un solo representante por clase de simetría

Piezas = una solución por cada grupo de 8 simétricas; al final de cada nivel se añaden las
simetrías del resultado.

| N | piezas | alcanzadas | niveles | tamaño de los niveles |
|---|---|---|---|---|
| 11 | las 8 simetrías | 752 (28 %) | 4 | 340, 196, 120, 76, 20 |
| 11 | un representante | 364 (14 %) | 1 | 340, 24 |
| 13 | las 8 simetrías | 73368 (99,5 %) | 7 | 2384, 6876, 13556, 24140, 21464, 4532, 384, 32 |
| 13 | un representante | 9656 (13 %) | 26 | 2384, 2004, 1228, 824, 664, 480, 296, … (embudo) |

Con un representante los niveles sí forman un embudo, pero la cobertura se hunde: las copias
giradas y reflejadas hacen falta como piezas (una misma solución aporta mitades distintas en
cada orientación).

---

## Predecir el total con la curva de niveles (`perfil.py`)

Niveles por filas: `P_k(N)` = colocaciones sin ataques de k reinas en las k primeras filas
(`P_1 = N`, `P_N = Q(N)`). Factor de cada nivel `b_k = P_{k+1}/P_k` = casillas libres de
media en la fila siguiente. `Q(N) = ∏ b_k`.

Ejemplo N=14: 1, 14, 156, 1364, 9632, 54068, 241484, 835056, 2211868, 4391988, 6323032,
6471872, 4511922, 1940500, **365596** (máximo en k=11; 27,4 millones de nodos en total).

**La curva es casi universal**: b_k/N en función de x = k/N.

| N | x=0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0,720 | 0,506 | 0,385 | 0,280 | 0,193 | 0,128 | 0,081 | 0,052 | 0,018 |
| 12 | 0,726 | 0,524 | 0,386 | 0,278 | 0,189 | 0,127 | 0,081 | 0,045 | 0,021 |
| 14 | 0,727 | 0,528 | 0,385 | 0,276 | 0,189 | 0,126 | 0,079 | 0,046 | 0,020 |

**Predicción** (`predict_log`): `ln(b_k/N) = G(x) + C(x)/N`, ajustado sólo con N = 10…14.

| N | predicho / real | N | predicho / real |
|---|---|---|---|
| 15 | 0,94 | 22 | 1,06 |
| 16 | 0,90 | 23 | 1,06 |
| 17 | 0,92 | 24 | 1,07 |
| 18 | 0,95 | 25 | 1,11 |
| 19 | 0,97 | 26 | 1,14 |
| 20 | 0,99 | 27 | 1,17 |
| 21 | 1,02 | | |

Trece valores, 12 órdenes de magnitud, todos dentro de ±17 %. Extrapolación a los no
conocidos (sin garantía; el sesgo crece ~3 % por N): Q(28) ≈ 3,0·10¹⁸, Q(29) ≈ 3,5·10¹⁹,
Q(30) ≈ 4,2·10²⁰.

Métodos peores: usar la curva de N=14 sin corrección (sobrestima ×1,1 en N=15, ×11,5 en
N=27); ajustar rectas en N punto a punto (inestable al final del tablero); ajustar
directamente ln Q/N − ln N con N ≤ 14 (×9,5 en N=27).

---

## Descomposiciones exactas de Q(N) en niveles (`particiones.py`)

Cada propiedad de una solución parte el conjunto y da una suma exacta. N=13 (Q = 73712):

- por columna de la primera reina: 2760 + 3799 + 5508 + 6023 + 7346 + 7385 + 8070 + (simétrico)
- por cortes: 130 + 26 + 56 + 340 + 788 + 2600 + 6756 + 20784 + 32044 + 10188
- por tramos: 130 + 18 + 48 + 112 + 212 + 1268 + 2708 + 8666 + 17608 + 25102 + 17840
- por simetría: 8×9197 + 4×32 + 2×4

**Forma normalizada por columna de la primera reina** (×N, en c/(N−1) = 0 … 0,5):

| N | 0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---|---|---|---|---|---|
| 12 | 0,42 | 0,71 | 1,02 | 1,22 | 1,38 | 1,39 |
| 13 | 0,49 | 0,73 | 1,01 | 1,20 | 1,30 | 1,42 |
| 14 | 0,46 | 0,71 | 0,98 | 1,21 | 1,35 | 1,42 |

Es estable ⇒ sirve de **frontera para la búsqueda**: con la forma media de N=12 y 13 se
predicen los 14 sumandos de N=14 con error medio del 1,4 % (máximo 3 %). Las de cortes y
tramos no son estables: su pico se desplaza con N (cortes: el 42 % está en J/(N−1) = 0,7 para
N = 11–12 y el 34 % en 0,8 para N = 14).

### Fronteras hasta N=18 (`primeras.c`, `fronteras.py`)

`./primeras N` (C, bits) cuenta las soluciones según las columnas de las dos primeras reinas.
N=17 en 45 s, N=18 en ~6 min; los totales coinciden con los conocidos.

**Cajones por la primera reina** — forma (×N) en c/(N−1):

| N | 0 | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 |
|---|---|---|---|---|---|---|
| 14 | 0,455 | 0,706 | 0,984 | 1,214 | 1,348 | 1,416 |
| 15 | 0,458 | 0,711 | 0,988 | 1,188 | 1,344 | 1,440 |
| 16 | 0,472 | 0,707 | 0,967 | 1,197 | 1,353 | 1,398 |
| 17 | 0,484 | 0,707 | 0,955 | 1,184 | 1,350 | 1,424 |

Ajuste simple (N = 16–18): **f(x) = 0,473 + 0,934·sin(πx)^1,198**, error cuadrático medio 0,010.

Reparto predicho con la forma media de N−1 y N−2 (total real):

| N | 1 reina: error medio / máximo | 2 reinas: cajones | error medio | cajones con error < 10 % |
|---|---|---|---|---|
| 14 | 1,4 % / 3,0 % | 156 | 11,7 % | 54 % |
| 15 | 2,2 % / 6,9 % | 182 | 9,9 % | 55 % |
| 16 | 0,9 % / 3,6 % | 210 | 7,3 % | 78 % |
| 17 | 1,5 % / 4,2 % | 240 | 6,5 % | 80 % |
| 18 | 0,6 % / 2,7 % | 272 | 5,8 % | 78 % |

**Todo predicho sólo con datos de N ≤ 14** (total por la curva de niveles × forma de N = 13–14):
los cajones de la primera reina salen entre 0,89 y 0,97 del valor real en N=17, y entre 0,92 y
1,01 en N=18.

---

## Regla aproximada del crecimiento de los niveles de fusión (`regla_niveles.py`)

Si ya se tienen H de las Q soluciones (tratadas como muestra al azar), T sale en el nivel
siguiente si para alguna plantilla sus dos mitades están presentes:

```
P(mitad presente) = 1 − (1 − H/Q)^c      c = nº de OTRAS soluciones con esa mitad
P(T en el nivel siguiente) = 1 − ∏_plantillas (1 − P_abierta · P_cerrada)
H' = H + (Q − H) · media_T [ P(T) ]
```

Versión "sólo números": c medio por tamaño de plantilla (N=13: mitad de 2 filas 573 y la de
11 filas 0,009; 3: 72,6 y 0,028; 4: 12,4 y 0,072; 5: 3,0 y 0,158; 6: 0,96 y 0,370).

| caso | real (acumulado) | regla detallada | sólo números |
|---|---|---|---|
| N=13, semilla reglas (2384) | 9260, 22816, 46956, 68420, 72952… → 7 niveles | 14611, 60634, 73469… → 5 | 16309, 73382 → 3 |
| N=13, semilla un tramo (130) | 366, 578, 770, 890… → 22 niveles | 249, 615, 2245, 13551, 59119… → 8 | 251, 628, 2364, 16120… → 6 |
| N=11, semilla reglas (340) | 536, 656, 732, 752 → se estanca en 28 % | llega a todo en 10 | en 3 |
| N=11, semilla un tramo (88) | 192, 280, 312, 320 → se estanca en 12 % | llega a todo en 12 | en 5 |

- La regla es **optimista**: supone que lo ya obtenido es una muestra al azar, y no lo es
  (las toroidales y sus descendientes comparten mitades entre sí mucho más que el promedio).
  Da un **mínimo de niveles** (5 ≤ 7, 8 ≤ 22) y el orden de magnitud del crecimiento.
- No predice el estancamiento de N=11.
- Primera versión (incluyendo a T en c) daba 2 niveles: error de planteamiento, corregido.

---

## Fórmulas combinatorias exactas para los primeros niveles (`formulas.py`, `perfil_k.c`)

`P_k(N)` = nº de formas de colocar k reinas sin ataques en las k primeras filas de un tablero
N×N (los niveles por filas; `P_N(N) = Q(N)`). Para k fijo son **polinomios en N de grado k**,
obtenidos por interpolación exacta y comprobados en todos los N restantes hasta 43:

| k | P_k(N) | vale para | comprobado |
|---|---|---|---|
| 1 | N | todo N | 34 valores |
| 2 | N² − 3N + 2 | todo N | 33 |
| 3 | N³ − 9N² + 30N − 36 | N ≥ 3 | 32 |
| 4 | N⁴ − 18N³ + 139N² − 534N + 840 | N ≥ 7 | 31 |
| 5 | N⁵ − 30N⁴ + 407N³ − 3098N² + 13104N − 24332 | N ≥ 11 | 27 |
| 6 | N⁶ − 45N⁵ + 943N⁴ − 11755N³ + 91480N² − 418390N + 870920 | N ≥ 17 | 20 |
| 7 | N⁷ − 63N⁶ + 1879N⁵ − 34411N⁴ + 417178N³ − 3336014N² + 16209916N − 36693996 | N ≥ 23 | 13 |

- Segundo coeficiente: 0, −3, −9, −18, −30, −45, −63 = **−3·k(k−1)/2** (cada par de filas
  prohíbe 3 casillas: misma columna y dos diagonales; el resto es inclusión–exclusión).
- Umbral de validez: 7, 11, 17, 23 para k = 4…7 (≈ 6k − 19): el polinomio vale cuando el
  tablero es ancho frente a las k filas. Para N menores hay correcciones de borde.
- Para N=13 dan exactos los niveles k ≤ 5: 13, 132, 1030, 6404, 31100 (coinciden con el árbol real).

---

## Las fórmulas de niveles como cota de la fusión de mitades (`medidor.py`)

Una mitad abierta de k filas consecutivas es una colocación parcial ⇒ **P_k(N) (polinomio
exacto) es el techo del nº de mitades distintas** y sirve de denominador para medir progreso:
`medidor_k = (mitades de k filas vistas en lo generado) / P_k(N)`.

**N=13** — cuántas mitades posibles se usan de verdad (U_k / P_k): k=1,2: 100 % · k=3: 99,8 %
· k=4: 94,5 % · k=5: 69,2 % · k=6: 35,9 % · k=7: 16,9 %.

Fusión con semilla de reglas: cobertura real 3,2 → 12,6 → 31,0 → 63,7 → 92,8 → 99,0 → 99,5 %;
medidor k=4: 26,8 → 58,5 → 78,6 → 90,0 → 93,8 → 94,4 → 94,4 % (se estabiliza en U_4/P_4 justo
cuando la fusión termina); medidor k=5: 6,3 → 18,9 → 35,7 → 55,7 → 66,6 → 68,8 → 69,0 %.

**N=17 a ciegas** (techos: P_3 = 2786, P_4 = 27020, P_5 = 216932, P_6 = 1448002), muestra de
3000 piezas por nivel:

| nivel | soluciones | medidor k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|
| 0 | 8620 | 78,0 % | 16,1 % | 2,5 % | 0,45 % |
| 1 | 42594 | 97,9 % | 45,8 % | 10,0 % | 1,97 % |
| 2 | 82730 | 99,6 % | 60,0 % | 15,8 % | 3,39 % |
| 3 | 113092 | 99,9 % | 68,4 % | 19,9 % | 4,41 % |
| 6 | 191892 | 100 % | 81,3 % | 29,2 % | 6,92 % |

El medidor de k=3 llega al 100 % (todas las 2786 aperturas de 3 filas aparecen); los de k=4–6
siguen subiendo: queda mucho por generar (192 mil de 95,8 millones).

## Nivel 1 numerado entre primos (`nivel1.py`, `nivel1_formas.py`)

Fusion de dos toroidales de un tramo (a1,c1)+(a2,c2) con plantillas de onda, p = 11, 13, 17, 19, 23.

- Quitando las fusiones triviales (1 fila cerrada devuelve la propia toroidal), quedan
  **104, 236, 748, 864, 1520** soluciones, y en el 100% las pendientes son **opuestas (a, -a)**:
  el nivel 1 es exactamente la regla de reflejos (restringida a plantillas de onda).
- Pendientes con a^2 = -1 (p=13: 5; p=17: 4) no dan ninguna fusion.
- Numeracion que relaciona primos: clase {a, -a, 1/a, -1/a}. Soluciones por clase:
  clase 2: 60, 120, 256, 308, 528 · clase 3: 44, 116, 376, 384, 656 · clase 4: -, -, 0, 100, 132 ·
  clase 5: -, 0, 116, -, 116 · clase 7: -, -, -, 72, 88.
- Filas cambiadas: siempre par. 2 filas: 20, 108, 324, 392, 904 · 4 filas: 84, 128, 408, 460, 592 ·
  6 filas: 0, 0, 16, 12, 20 · 8 filas: solo 4 en p=23.
- Formas: unas pocas son locales y se repiten igual (a=3: centro ±1, ±2 en los cinco primos;
  a=2: centro ±1, ±3 desde p=17); la mayoria se mide desde el borde y escala con p
  (a=2: dos filas a distancia p-3 en los cinco primos).

## Niveles 2 y 3 desde solo toroidales, con cotas (`nivel2.py`)

Semilla = toroidales de un tramo, sin cerrar por simetrias, plantillas de onda.

| p | nivel 0 | 1 | 2 | 3 | cota k=3 (niveles 0-3) | cota k=4 (niveles 0-3) |
|---|---|---|---|---|---|---|
| 11 | 88 | 192 | 280 | 312 | 16, 27, 37, 38 % | 3.6, 6.2, 9.2, 10.3 % |
| 13 | 130 | 366 | 558 | 698 | 13, 23, 30, 35 % | 2.0, 4.4, 6.0, 7.4 % |
| 17 | 238 | 986 | 2330 | 4162 | 8.5, 21, 34, 48 % | 0.9, 2.5, 4.8, 7.8 % |
| 19 | 304 | 1168 | 2772 | 5582 | 7.3, 17, 28, 43 % | 0.6, 1.7, 3.2, 5.7 % |

- La cota k=2 esta llena (98-100%) desde la semilla; k=3 y k=4 quedan muy lejos.
- Nivel 2: el 89-100% de las nuevas sigue usando solo un par de pendientes opuestas (a, -a)
  (p=11: 56/88; 13: 192/192; 17: 1248/1344; 19: 1428/1604). Nivel 3: 71-78% (p=11: 25%).
- Piezas del nivel 2: (toroidal + nivel 1) y (nivel 1 + nivel 1) a partes parecidas.

## Funcion de crecimiento desde las toroidales (`crecimiento.py`)

Hipotesis: nuevas(n+1) = beta * H(n)^2 * (1 - H(n)/Q), H = acumulado (con simetrias).

- N=13: nuevas/H^2 (x1e-4) por nivel: 140, 15.8, 5.8, 2.0, 1.4, 0.96, 1.26, 1.18, 1.09, 0.87, 0.80,
  0.89, 1.37, 1.50, 1.34, 1.08 -> desde el nivel 4 es casi constante, beta = 1.2e-4.
  Simulado desde el nivel 4 (890): 984 1098 1240 1422 1659 1981 2439 3128 4250 6290 10625 22198 63450
  Real:                            1002 1098 1250 1434 1658 1898 2186 2610 3542 5418 9356 18844 39188 63312
  Desierto ~ 1/(beta*H) = 9.4 niveles. El final real es mas lento (el freno 1-H/Q se queda corto).
- N=11: la razon no se estabiliza (134, 24, 4.1, 0.8) y el proceso se para en 320.
- p=17 (5 niveles): H = 238, 986, 2370, 4478, 7958, 15506; razon 132, 14.2, 3.75, 1.74, 1.19.
  Sin desierto: en el nivel 4 beta*H ya es ~1. Los primeros valores de la razon casi coinciden
  con los de N=11 y 13 (~135, ~15-24, ~4-6).

### Mas niveles en p=17 y 19: la ley es H^1.5, no H^2

p=17: H = 238, 986, 2370, 4478, 7958, 15506, 36306, 112788, 525388 (8 niveles)
p=19: H = 304, 1168, 2824, 5984, 12028, 26832, 70520, 245064 (7 niveles; parado a mano)

- nuevas/H^2 NO es constante: p=17 baja 1.74, 1.19, 0.87, 0.58, 0.32 (x1e-4); p=19 igual.
- nuevas/H^1.5 SI: p=17 niveles 3-7 = 0.0116, 0.0106, 0.0108, 0.0111, 0.0109 (exponente local 1.5).
  p=19 niveles 3-6 = 0.0131, 0.0112, 0.0099, 0.0093 (exponente local 1.28, 1.35, 1.43, subiendo).
  N=13 en la explosion (niveles 14-16, con freno) = 0.0107, 0.0120, 0.0106; en el desierto 0.003-0.004.
- Ley comun de la fase explosiva: nuevas ~ 0.0105 * H^1.5 * (1 - H/Q).
  Simulado desde el nivel 3: p=17 -> 7624, 14614, 33161, 96544, 411202 (real 7958, 15506, 36306, 112788, 525388);
  p=19 -> 10844, 22702, 58618, 207633 (real 12028, 26832, 70520, 245064).

### Fusion en C con hilos (`crecimiento.c`) y exponente medido

`cc -O3 -o crecimiento crecimiento.c -lpthread; ./crecimiento N hasta hilos`. Reproduce exactamente
los niveles de Python (N=13 completo en 5 s: 73344). Datos en data/crecimiento_c_N.txt.

- p=17: 238, 986, 2370, 4478, 7958, 15506, 36306, 112788, 525388, 4246790, 48296008 (nivel 10 = 50.4% de Q)
- p=19: 304, 1168, 2824, 5984, 12028, 26832, 70520, 245064, 1275208, 11438256
- p=23: 460, 1980, 6396, 18272, 55396, 198444, 947392, 6737652

Exponente local (nuevas ~ H^e): p=17: 1.35, 1.52, 1.53, 1.49, 1.43 (niveles 4-8);
p=19: 1.28, 1.35, 1.43, 1.43, 1.39; p=23: 1.09, 1.22, 1.30, 1.31. No es 1.618 (con ese exponente la
constante cae nivel a nivel en los tres). Queda entre 1.3 y 1.5 y parece bajar con p.

## Genealogia completa aplanada (`genealogia.py`) y clase "2 rectas" (`lineas2.py`)

Toda solucion alcanzada por fusion desde toroidales es "a trozos": cada fila esta sobre la recta
(a, c) de la hoja de la que viene. `data/genealogia_N13.npz` guarda nivel, piezas, plantilla y la
recta de cada fila para las 73344 alcanzadas (85 s).

- Rectas distintas por solucion (no minimo garantizado): nivel 1 = 2; nivel 2 = 2.9; nivel 3 = 4.2;
  nivel 4 = 5.9; desde el nivel 7 entre 7 y 8.6 (de 13 filas). Total: 1:130 2:612 3:844 4:1322
  5:2668 6:5512 7:10332 8:15604 9:15516 10:11876 11:6024 12:2476 13:428.
- El nivel de fusion NO ordena por sencillez: hay soluciones de 2 rectas en los niveles 11-21.

Clase "2 rectas" exhaustiva (cualquier reparto de filas, sin plantilla de onda), no toroidales:
p=11: 104 · p=13: 2794 · p=17: 9324 · p=19: 18132.
Pendientes: p=13 opuestas 1942, inversas 820, otra 32; p=17 opuestas 6890, inversas 2314, otra 120;
p=19 opuestas 7280, inversas 5068, otra 5784; p=11 solo opuestas.
En N=13 la fusion por niveles solo saca 236 de ellas en el nivel 1; el resto aparece disperso
hasta el nivel 21 y 16 nunca.

## Minimo de rectas por solucion (`minrectas.c`, exhaustivo)

Minimo n de rectas toroidales (pendiente 2..p-2) que cubren las p reinas. Todas las soluciones.
`./minrectas p hilos` -> data/minrectas_p{p}.txt; `./minrectas_azar p 1 n` = modelo nulo (permutaciones al azar).

| m rectas | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| p=11 | 88 | 104 | 912 | 1576 | 0 | 0 |
| p=13 | 130 | 2794 | 6240 | 48896 | 15652 | 0 |
| p=17 | 238 | 9324 | 52820 | 1805790 | 62322726 | 31624206 |

Azar (% de permutaciones): p=11 2:0.04 3:6.3 4:80.7 5:11.6 6+:1.3 · p=13 3:0.4 4:34 5:64 6+:1.7 ·
p=17 4:0.2 5:29.6 6:69.8 7+:0.4. Las soluciones bajan ~1 recta respecto al azar; las clases 2 y 3
estan muy enriquecidas (p=13: 3.8% y 8.5% frente a 0.0005% y 0.4%). Cuidado: con p-3 pendientes,
cualquier permutacion se cubre con pocas rectas; el n de rectas por si solo no es estructura.
La genealogia de fusion (7-9 rectas) estaba lejos del minimo.

Pendientes de una cobertura minima: m=2 siempre opuestas en p=11; p=13 opuestas 1942, inversas 820,
otra 32; p=17 opuestas 6890, inversa 1156, -inversa 1158, otra 120. m=3 dominante: (a, a, -a)
(dos paralelas + una opuesta): p=13 2620/6240, p=17 33256/52820.

## ¿La clase m se construye con las menores? (`recursion_clases.py`, `recursion_nulo.py`)

Clase m = n minimo de rectas (minrectas.c; DUMPMAX=k ./minrectas p vuelca data/clases_p{p}.txt).
Para T de clase m: (a) A de clase < m + cambiar las filas de una recta; (b) T = A en unas filas y B
en el resto, A y B de clase < m, reparto libre; (c) hasta tres piezas de clase < m.

| | clase 2 | clase 3 | clase 4 | clase 5 |
|---|---|---|---|---|
| p=11 (a/b/c) | 100/100/100 | 8.8/8.8/100 | 8.6/33/100 | – |
| p=13 | 100/100/100 | 73/80/100 | 8.1/48/100 | 39/99.9/100 |
| p=17 (muestra 1000) | 100/100/100 | 75/78/100 | 3.9/16/99.9 | – |

Modelo nulo (permutaciones al azar contra las mismas piezas): dos piezas 0-12%, tres piezas 20-82%.
-> (b) si discrimina; (c) es casi automatico y no prueba nada por si solo.

## Clase 2 exacta con las espirales de super_queens (`espirales.py`)

Espiral (d, sentido, giro i) = recta T(r) = s*2d*(r+1) - i (mod p) (numeracion de init_sols_label).
Mezclar A y B: las filas que toman B deben ser union de CICLOS de pi(r) = A^-1(B(r)) = m*r + k,
m = pendiente_B/pendiente_A. Mismo sentido y velocidad (m=1): un ciclo -> nunca. Sentido contrario y
misma velocidad (m=-1): parejas de filas simetricas respecto a la fila de cruce. En general
(p-1)/ord(m) ciclos de longitud ord(m). Solo queda comprobar diagonales: recuento exacto.

| p | clase 2 | m=-1 (pares utiles / sol.) | otros ordenes (sol.) |
|---|---|---|---|
| 11 | 104 | 68 / 104 | - |
| 13 | 2794 | 585 / 1942 | orden 3: 820, orden 4: 32 |
| 17 | 9324 | 1331 / 6890 | orden 4: 1278, orden 8: 1156 |
| 19 | 18132 | 1684 / 7280 | orden 9: 5776, orden 3: 5076 |
| 23 | 27348 | 3254 / 10420 | orden 11: 16928 |

Orden (p-1)/2 (dos ciclos = residuos / no residuos cuadraticos): exactamente 2 soluciones por par util;
pares utiles 2*17^2, 8*19^2, 16*23^2 (p=17, 19, 23). Ejemplo p=13: espirales 15 = (1,-,7) y 22 = (1,+,11)
se cruzan en la fila 0; ciclos {1,12},{2,11},...,{6,7}; 2 de las 62 uniones dan solucion.

### ¿Las parejas de espirales se trasladan entre N? (`espirales_transfer.py`)

- Traslado literal (mismas velocidades, sentidos y giros) o por casilla de cruce, de p a p': igual que
  la tasa base (13->17: 38.6% vs 38.4%; 17->19: 37.8 vs 36.2; 19->23: 33.2 vs 32.9). No se traslada.
- Lo que manda es el TIPO (velocidad, sentido) x 2. Cada tipo es 0%, 100% (todas las p^2 combinaciones
  de giros funcionan) o intermedio. Tipos 100%: p=13 v1+-, v3+-, v4+-, v2-v5-, v2+v5+; p=17 v2+-, v3+-,
  v7+-, v1-v4+, v1+v4-, v5-v6+, v5+v6-; p=19 12 tipos; p=23 16 tipos.
- En los tipos 100% casi todas las soluciones son TOROIDALES (13: 2366/2630, 17: 7514/8034,
  19: 14440/15544, 23: 16928/16928): son las toroidales de dos clases (residuos cuadraticos, clases_primos).
  En los tipos que dependen del giro ninguna es toroidal (0/164, 0/1290, 0/2588, 0/10420).
- Conclusion: la regla "velocidad x contra velocidad y, para cualquier giro" existe y es exacta
  justo para las mezclas toroidales; las no toroidales dependen del borde (giro) y no se trasladan.

### Formula de las parejas de velocidades "siempre" (`formula_espirales.py`)

Pendientes alfa = s1*2d1, beta = s2*2d2. H = subgrupo de Z_p* generado por
m = beta/alfa, (alfa+1)/(beta+1), (alfa-1)/(beta-1).
**La pareja funciona con cualquier giro <=> H != Z_p*.** Soluciones toroidales por pareja de giros:
2^k - 2, k = (p-1)/|H| (filas = uniones de clases de H medidas desde la fila de cruce).
(Deduccion: columnas siempre cuadran; sumas y restas exigen que el conjunto de clases elegido sea
invariante por (alfa+1)/(beta+1) y (alfa-1)/(beta-1).)

Comprobado exacto (tipos y recuentos) en p = 11, 13, 17, 19, 23: 0, 2366, 7514, 14440, 16928 toroidales
de dos rectas; p=29, 31 por muestreo (222 pruebas, 0 fallos); predice 166518 y 230640.
Soluciones NO toroidales de clase 2: solo en tipos con ord(m) = 2, 3 o 4 (p <= 23):
13: {2: 252, 3: 144, 4: 32}; 17: {2: 1688, 4: 122}; 19: {2: 2948, 3: 744}; 23: {2: 10420}.
-> clase 2 sin comparar todos con todos: formula para las toroidales + probar solo parejas con ord(m) <= 4.

### Tabla clase 1 / clase 2 (`tabla_clases.py`, todas las parejas probadas)

| p | clase 1 (lineales) | clase 2 toroidales (formula = probado) | clase 2 no toroidales | toroidales totales (A007705) |
|---|---|---|---|---|
| 11 | 88 | 0 | 104 | 88 |
| 13 | 130 | 2366 | 428 | 4524 |
| 17 | 238 | 7514 | 1810 | 140692 |
| 19 | 304 | 14440 | 3692 | 820496 |
| 23 | 460 | 16928 | 10420 | 128850048 |
| 29 | 754 | 166518 | 35084 | ? |
| 31 | 868 | 230640 | 75872 | ? |

La formula de las toroidales de clase 2 acierta exacto hasta p=31. La regla "no toroidales solo con
ord(m) <= 4" FALLA: p=29 tiene 112 con ord 7; p=31 tiene 1584 con ord 5 y 96 con ord 6.
No toroidales por ord(m): 29 {2: 31148, 4: 3824, 7: 112}; 31 {2: 57744, 3: 16448, 5: 1584, 6: 96}.

## Toroidales por clase (min rectas) y la clase 3 toroidal

`TORO=1 ./minrectas p` recorre solo las toroidales (totales = A007705 exacto):

| p | toroidales | clase 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 13 | 4524 | 130 | 2366 | 2028 | - | - | - |
| 17 | 140692 | 238 | 7514 | 27744 | 35836 | 46240 | 23120 |
| 19 | 820496 | 304 | 14440 | 0 | 51984 | 207936 | 545832 |

Todas son multiplo de p^2 (traslaciones). En las clases >= 3 las rectas NO pasan por un punto comun
(`radial.py`: clase 3 de 13 necesita 5 pendientes desde cualquier reina).

**Clase 3 toroidal = pendientes (a, a, -a) con a^2 = -1 mod p** (dos paralelas y una opuesta), en
13 y 17 todas; por eso p = 19 (3 mod 4) no tiene ninguna. `clase3_toro.py` las cuenta combinando
esas tres rectas: 2028, 27744 (= minrectas); predice 4167996 (p=29), 58845096 (37), 199232120 (41)
(sin comprobar que ninguna baje a clase 2 en esos p).

## Toroidales de clase >= 3: simetrías, regla exacta de la clase 3 y tipos de las clases 4–6

Scripts nuevos: `orbitas.py`, `ver_orbitas.py`, `clase3_filas.py`, `clase3_regla.py`, `haz_i.py`,
`clase_m.py`, `tipos.py`, `radial_impar.py`, `vecinos.py`, `minrectas_toro.c`.
Marcas: **[P]** = probado con álgebra, **[E]** = comprobado exhaustivamente en los p indicados,
**[O]** = solo observado.

### 1. Hay muy pocas toroidales "distintas": el grupo de simetrías completo (`orbitas.py`)

Además de las p² traslaciones, el toro tiene 8(p−1) simetrías que llevan toroidales a toroidales:
multiplicar todo el tablero por u (p−1 formas) y 8 "giros" que permutan las cuatro direcciones
prohibidas (filas, columnas y las dos diagonales). Uno de ellos es (r,c) → (r+c, c−r), que en el
toro es un giro de 45° (en el tablero plano no existe). Sobre la pendiente a de una recta los giros
actúan como a → −a, 1/a, (a−1)/(a+1) y sus combinaciones. **[P]**

Órbita = conjunto de soluciones que se obtienen unas de otras con estas simetrías. **[E]**

| p | órbitas de traslación | órbitas del grupo completo | por clase (tamaños de órbita, en órbitas de traslación) |
|---|---|---|---|
| 13 | 36 | 5 | c1: 2+8 · c2: 6+8 · c3: 12 |
| 17 | 500 | 23 | c1: 2+4+8 · c2: 2+4+4+8+8 · c3: 16+16+32+32 · c4: 4+8+16+16+16+32+32 · c5: 32+128 · c6: 16+64 |
| 19 | 2288 | 23 | c1: 8+8 · c2: 16+24 · c4: 144 · c5: 72×6+144 · c6: 72+144×10 |

Esto explica los números "por órbita" (26, 96, 124, 160, 80; 40, 0, 144, 576, 1512): son sumas de
tamaños de órbita, y cada tamaño es un divisor de 8(p−1) (128 en p=17, 144 en p=19). Ejemplo: toda la
clase 4 de p=19 es UNA sola solución movida por las 144 simetrías; la clase 6 de p=19 son 11.
`data/orbitas_p{p}.json` guarda un representante de cada una.

Las pendientes se agrupan igual. Pendientes especiales (fijas por algún giro):
- **i** con i² = −1 (existe si p ≡ 1 mod 4): {i, −i}. Fijas por el giro de 45°.
- **s** = ±1 ± √2 (existe si p ≡ ±1 mod 8): 4 pendientes. p=17: {5,7,10,12}; p=23: {4,6,17,19}.
- el resto va en grupos de 8. (p=13: 2+8; p=17: 2+4+8; p=19: 8+8; p=23: 4+8+8.)

### 2. Clase 3: regla exacta de las filas (`clase3_filas.py`, `clase3_regla.py`)

Rectas: B de pendiente −a y dos paralelas A1, A2 de pendiente a, con a² = −1.
μ4 = {1, a, −1, −a} (las cuatro raíces cuartas de 1). "Clase de x" = el conjunto x·μ4 (4 filas);
hay k = (p−1)/4 clases. Sean r1 y r2 las filas donde B corta a A1 y a A2.

**Regla.** A1 se queda con clases enteras de filas medidas desde r1; A2 con clases enteras medidas
desde r2; B con todas las demás filas. Si X e Y son los conjuntos de clases de A1 y A2 y se
normaliza r1 = 0, r2 = 1:

    filas de A1 = X·μ4          filas de A2 = 1 + Y·μ4          B = el resto
    condición única: no hay x en X·μ4, y en Y·μ4 con x − y en μ4.

**Fórmula.** nº de toroidales de clase 3 de este tipo (HAZ 2+1 sobre i) = p² · (p−1) · N(p), con N(p) = nº de pares
(X, Y) no vacíos que cumplen la condición. Se calcula al instante (grafo de k vértices: la clase C
es vecina de la clase de x−1 para cada x de C; se cuentan pares de conjuntos sin aristas entre sí).

| p | 13 | 17 | 29 | 37 | 41 | 53 | 61 | 73 |
|---|---|---|---|---|---|---|---|---|
| N(p) | 1 | 6 | 177 | 1194 | 2963 | 41506 | 230189 | 2887557 |
| total HAZ (i) | 2.028 | 27.744 | 4.167.996 | 58.845.096 | 199.232.120 | 6.062.698.408 | 51.391.996.140 | 1.107.920.970.216 |

Qué está comprobado:
- **[P]** Que las soluciones construidas así son toroidales (es reescribir "no se atacan": dos reinas
  en rectas de pendientes a y −a que se cortan en r0, en filas r y r', no se atacan ⇔ (r'−r0)/(r−r0)
  no está en μ4; y dos reinas en las paralelas no se atacan ⇔ x − y no está en μ4).
- **[E]** Que son todas las de pendientes (a, a, −a): p=13 y 17 coincide con el recorrido completo; p=29, 37, 41 coincide con la
  búsqueda fila a fila de `clase3_toro.py` (que no supone clases enteras). No está probado para todo p
  que las clases tengan que ir enteras.
- **[E]** Ninguna baja a clase 2 (p ≤ 41) y son todas distintas (p ≤ 29).
- **[E] CORRECCIÓN: (a, a, −a) NO es toda la clase 3 a partir de p=29.** `clase3_todas.py` prueba todas
  las ternas de rectas sin suponer pendientes (valida 12, 96, 0 en p=13, 17, 19; `data/clase3_todas.txt`):

  | p | clase 3 total (órbitas de traslación) | HAZ 2+1 | tres pendientes distintas |
  |---|---|---|---|
  | 23 | 0 | 0 | 0 |
  | 29 | 5.276 (= 4.437.116 soluciones) | 4.956 sobre (i, −i) | 320: 224 en triángulo (i, −i, x) + 96 por un punto |
  | 31 | 1.212 (= 1.164.732 soluciones) | 960 sobre (w, w²) y sus 3 pares simétricos | 252, todas por un punto (radiales de 3 pendientes) |

  Es decir: la predicción 4.167.996 para p=29 era solo la parte HAZ (94 %), y "p ≡ 3 mod 4 no tiene
  clase 3" es falso desde p=31, donde el haz usa las raíces cúbicas de 1 (w=5, w²=25; H = μ6, 5 clases).
- **[E] La regla vale para cualquier par de pendientes (α, β)**: clases enteras de H(α,β) desde cada
  corte, y condición "ningún x − y en E(β,α)". Órbitas de traslación con patrón (α, α, β) =
  (p−1)·N(α,β)/2. Da 2.478 para (12,12,17) en p=29 y 120 para (5,5,25) en p=31, igual que la búsqueda
  completa; da 0 en p=13, 19 para (w, w²) (pocas clases) y 0 para los pares s de p=17, 23.
  Predice para (w, w, w²): 450 en p=37 y 1.386 en p=43 (sin comprobar).
- **[O]** N(p) del par (i, −i) crece ≈ ×2,3 por cada clase que se añade (p → p+4).

Por qué a² = −1: con las coordenadas u = c − a·r, v = c + a·r las rectas de pendiente ±a son las
verticales y horizontales, y las cuatro direcciones prohibidas pasan a ser las pendientes
{1, −1, a, −a} = μ4. Una toroidal es entonces un conjunto de p puntos donde (Δv)⁴ ≠ (Δu)⁴ para todo
par: por eso aparecen las clases de μ4. **[P]**

La misma cuenta general: para dos pendientes α, β el conjunto de cocientes prohibidos es
E(α,β) = {1, α/β, (α+1)/(β+1), (α−1)/(β−1)} y H = subgrupo generado por E (el de la fórmula de la
clase 2). Para (i, −i) E ya es un grupo (μ4): es el caso más limpio.

### 3. Clases 4, 5 y 6: tipos de cobertura (`tipos.py`, `clase_m.py`, `data/tipos.txt`)

Mirando TODAS las coberturas mínimas de cada órbita salen dos tipos con estructura y un resto:

- **HAZ n+m**: solo dos pendientes; n paralelas de una y m de la otra.
- **RADIAL**: todas las rectas pasan por una misma reina (familia ciclotómica con centro).
- **otra**: ni lo uno ni lo otro.

p=17 (órbitas de traslación; pendientes i = {4,13}, s = {5,7,10,12}):

| clase | tipo | cuántas |
|---|---|---|
| 3 | HAZ 2+1 sobre (i, −i) | 96 (todas) |
| 4 | HAZ 3+1 sobre (i, −i) | 48 |
| 4 | RADIAL de 4 pendientes: (s,s,s,s) 28, (i,−i,s,s) 16, (2,3,7,12) y simétricas 32 | 76 |
| 5 | otra (3 y 5 coberturas mínimas) | 160 |
| 6 | otra (17 y 286 coberturas mínimas) | 80 |

p=19 (no hay i ni s; w = 7, w² = 11 son las raíces cúbicas de 1):

| clase | tipo | cuántas |
|---|---|---|
| 4 | HAZ 2+2 sobre (w, w²), |H| = 6 | 144 (todas, una sola solución salvo simetría) |
| 5 | HAZ 3+2 (pendientes corrientes) 144; RADIAL de 5 pendientes 72; otra 360 | 576 |
| 6 | HAZ 3+3 72; otra 1440 (entre 15 y 289 coberturas mínimas cada una) | 1512 |

Lecturas:
- **[E]** Las coberturas de 3 y 4 rectas son casi siempre únicas y usan pendientes especiales (i, s, w).
  En clase 6 hay decenas o cientos de coberturas distintas con pendientes cualesquiera: ahí "6 rectas"
  ya no describe nada (una permutación al azar de 17 también se cubre con 5–6 rectas).
- **[E]** HAZ 3+1 sobre (i, −i) sigue la misma regla que la clase 3 (`haz_i.py`): cada paralela j toma
  clases enteras de μ4 desde su corte r_j con B, y para cada par j, l no hay x, y con
  x − y en (r_l − r_j)·μ4. Da 0 en p=13 y 48 en p=17 (= lo observado). Predice 31.276 órbitas de
  traslación en p=29 (sin comprobar, y sin comprobar que no bajen de clase). Con 4+1 en p=17 la regla
  da 4 soluciones que en realidad son de clase 4: a partir de ahí el haz deja de ser cobertura mínima.
- **[E]** RADIAL = `radial_impar.py`: T(r) = a_j·r según la clase de r (parejas ±r), todas las rectas
  por el origen. Por nº de pendientes: p=13 {1:10, 2:14, 5:12} (son las 36: en p=13 TODAS las
  toroidales tienen centro de simetría); p=17 {1:14, 2:26, 4:76, 5:32, 6:16, 8:16};
  p=19 {1:16, 2:24, 5:72, 6:72, 7:288, 8:72}; p=23 {1:20, 4:88, 6:1056, 7:968, 8:2244, 9:2024, 10:924, 11:132}.
  Los 76 de p=17 son exactamente la clase 4 radial.
- **[O]** En las RADIAL de 4 pendientes siempre hay una pareja s de la forma (a, −1/a) con a+(−1/a) = ±2
  (p=17: (7,12) o (5,10); p=23: (6,19) o (4,17)) que se lleva casi todas las filas.

### 4. Relación entre clases (`vecinos.py`, `data/vecinos_p*.txt`)

Pregunta: ¿una toroidal de clase m sale de otra de clase m−1 cambiando solo las reinas de una recta
de su cobertura mínima? Se liberan las filas que solo cubre esa recta y se buscan todas las demás
toroidales que coinciden en el resto. Azar = liberar el mismo número de filas elegidas al azar.

p=17 (órbitas de traslación con alguna vecina de clase m−1; real | azar):

| clase | real | azar | comentario |
|---|---|---|---|
| 2 | 26/26 | 1/26 | trivial: se endereza la segunda recta |
| 3 | 96/96 | 4/96 | toda clase 3 viene de una clase 2 |
| 4 | 48/124 | 0/124 | justo las 48 de HAZ 3+1; las 76 RADIAL no tienen ninguna vecina |
| 5 | 0/160 | 0/160 | aisladas |
| 6 | 0/80 | 0/80 | aisladas |

p=13: clase 3 → 2 en 12/12 (azar 0/12). **[E]** Conclusión: la escalera 1 → 2 → 3 → 4 existe solo
dentro de la familia HAZ sobre (i, −i) (añadir una paralela más cada vez). Las RADIAL y las clases 5–6
no se alcanzan cambiando una recta.

### 5. Comprobación en p=23 (`minrectas_toro.c`, `tipos23.py`, `data/run_p23.log`)

`CAP=7 DUMPMAX=5 ./minrectas_toro 23 10` (70 min, 10 hilos; fija T(0)=0 y multiplica por p).
Total 128.850.048 = A007705. **[E]**

| clase | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 o más |
|---|---|---|---|---|---|---|---|---|
| soluciones | 460 | 16.928 | 0 | 209.484 | 1.350.008 | 28.443.272 | 98.597.136 | 232.760 |
| por órbita de traslación | (20) | 32 | 0 | 396 | 2.552 | 53.768 | 186.384 | 440 |

Predicciones hechas antes de ver el resultado:
- clase 1 = p(p−3) = 460 y clase 2 = 16.928: **aciertan**.
- clase 3 = 0 (no hay i ni w en p=23): **acierta** (confirmado además con `clase3_todas.py`).
- clase 4 = solo las 88 radiales impares de 4 pendientes: **falla**. Las 88 están (son exactamente
  las RADIAL), pero hay 396. La clase 4 son 5 soluciones salvo simetría (`data/tipos_p23_clase4.txt`):

| órbitas de traslación | tipo |
|---|---|
| 88 (44+44) | RADIAL de 4 pendientes (s, s, x, y); 44 de ellas admiten también HAZ 3+1 sobre (s, s) |
| 44 | HAZ 3+1 sobre un par s (pendientes ±1±√2), sin centro |
| 88 | 2 paralelas + 2 rectas sueltas, pendientes corrientes |
| 176 | 4 pendientes distintas corrientes que no pasan por un punto |

Lectura: las familias HAZ y RADIAL se trasladan entre primos (aparecen en 17, 19 y 23 con las
pendientes especiales que existan en cada uno), pero no agotan la clase 4: en p=23 dos de las cinco
soluciones (264 de 396) usan pendientes corrientes sin centro común, y de esas no hay regla todavía.
La clase "típica" sube con p: 5–6 en p=17, 6 en p=19, 7 en p=23.

## Niveles 0 y 1 definidos, haces n+m, y el mapa completo de p=17 y p=19 (octubre 2026)

Scripts nuevos: `espirales_niveles.py`, `espirales_log.py`, `dos_polos.py`, `cilindro.py`, `haces.py`,
`haces.c` (+ `haces_run.sh`, `haces_resumen.py`), `polinomios.py`, `semillas.py`, `intercambios.py`,
`mezclas.py`, `mezclas3.py`, `mezclas_porque.py`, `mezclas_niveles.py`, `nueve.py`, `parecidos.py`.
Marcas: **[P]** probado, **[E]** exhaustivo en los p indicados, **[O]** observado.
Aviso de nombres: aquí "nivel 0 / nivel 1" = clase 1 / clase 2 toroidal (una y dos espirales), no los
niveles de fusión de `nivel1.py`.

### 1. Nivel 0 y nivel 1: forma cerrada

- **Nivel 0** [P]: T(r) = a·r + c, a en 2..p−2. Recuento p(p−3). Clásico (Pólya 1918; soluciones "regulares").
- **Nivel 1** [P]: pendientes α ≠ β que se cruzan en (r0, c0); T(r) = c0 + α(r−r0) si r−r0 ∉ S,
  c0 + β(r−r0) si r−r0 ∈ S. H = ⟨β/α, (α+1)/(β+1), (α−1)/(β−1)⟩; S = unión de clases de H, ni vacía ni
  total. Total = p² · Σ_{α<β} (2^k − 2), k = (p−1)/|H|. Es "si y solo si" y la cobertura es única.
  Fuerza bruta p=13, 17: 14 y 26 por traslación ✓.

| p | 13 | 17 | 19 | 23 | 29 | 31 | 37 | 41 | 43 |
|---|---|---|---|---|---|---|---|---|---|
| nivel 1 / p² | 14 | 26 | 40 | 32 | 198 | 240 | 958 | 1418 | 832 |

  (37, 41, 43: fórmula, sin recorrido exhaustivo.) Literatura: k=2 = quadratic (strong) orthomorphisms;
  k general = cyclotomic orthomorphisms con dos multiplicadores (Evans; Fear–Wanless arXiv 2101.00859).
  No encontré publicado un recuento exacto de las de dos pendientes; lo conocido es asintótico (Bell).

### 2. Cómo se ven (figuras)

- `fig_niveles_espirales_p{p}.png`: rejilla polar normal; nivel 1 = X de dos espirales.
- `fig_espirales_log_p{p}.png`: **rejilla de potencias** (cruce al centro; anillos y rayos reordenados por
  potencias de una raíz primitiva g). Ahí las dos espirales son LA MISMA espiral de una vuelta girada
  log(β)−log(α) rayos, y las reinas se turnan con periodo k. [P]; fuerza bruta p=13, 17: todos los
  turnos periódicos (13: periodo 2 ×8, 3 ×6; 17: periodo 2 ×14, 4 ×12).
  Vale para cualquier solución con todas las rectas por una reina (RADIAL); no para varios cruces.
- `fig_cilindro_p{p}.png`: fila = altura, columna = ángulo; rectas = hélices, paralelas = misma hélice girada.

### 3. Clase 3 y haces n+1: regla de "reclamos" (`dos_polos.py`)

Reescritura de la regla de la clase 3 [P]: cada paralela reclama clases enteras de H alrededor de SU
cruce con la recta suelta; la única condición es que **los reclamos no se pisen** (ninguna fila en dos
reclamos); la suelta toma el resto. En tabla: fila r ↦ (clase de r−r1, clase de r−r2); A1 = columnas
enteras, A2 = renglones enteros, casillas X×Y vacías (la tabla son los números ciclotómicos de orden k).
Reproduce N(p) = 1, 6, 177, 1194, 2963 y los volcados (156/156 en p=13, 1632/1632 en p=17).
Con n paralelas: 12, 96, 4956 (2+1) y 0, 48, 31276 (3+1) en p = 13, 17, 29.
Probado y fallido: un solo turno con los dos cruces como polos, u = (r−r1)/(r−r2): sin periodicidad.

### 4. Censo de haces n+m sin suponer regla (`haces.py`, `haces.c`)

Busca todas las toroidales que en cada fila toman la reina de una de n rectas de pendiente a y m de
pendiente b, todas las parejas (a,b). Órbitas de traslación (con multiplicidad si una solución admite
dos haces: p=17 3+1 da 80 por 64 distintas; p=23, 132 por 88):

| p | 2+1 | 3+1 | 2+2 | 3+2 | 4+1 |
|---|---|---|---|---|---|
| 17 | 96 | 64 | 0 | | |
| 19 | 0 | 0 | 144 | 144 | 216 |
| 23 | 0 | 88 | 0 | | |
| 29 | 4.956 | 31.276 | 8.400 | | |
| 31 | 960 | | 1.200 | | |
| 37 | 46.584 | 655.884 | 164.160 | | |
| 41 | 119.880 | 2.536.680 | 584.320 | | |
| 43 | 11.088 | 52.864 | 19.488 | | |

- [E] Predicciones acertadas: p=29 3+1 = 31.276; p=37 tipo i 42.984 y tipo w 450/pareja; p=41 tipo i
  118.520; p=43 tipo w 1.386/pareja. Todas las soluciones n+1 halladas (p=29, 37, 41, 43) cumplen
  "clases enteras desde cada cruce".
- [E] Corrección: los ingredientes no son solo i, w, s. Sirve cualquier pareja con H pequeño; en p=41
  aparecen familias con |H|=8 (pendientes 2, −2 y giros; 960 en 2+1, 1.600 en 2+2) y |H|=10 (400; 320).
- [E] **En 2+2 la regla de clases enteras falla siempre** (p=19, 29, 37: 0 de 2.736, 1.200, 18.240).
  2+2 sobre (i,−i): 0 en 13 y 17; 8.400, 158.112, 582.400 en 29, 37, 41. Sobre (w,w²): 144 (p=19), 1.200 (p=31).
- Huecos entre primos: i existe si p≡1 (4), w si p≡1 (3), s si p≡±1 (8), y además hacen falta
  suficientes clases (13 y 19 tienen w pero 2+1 da 0).
- Datos: `data/haces_base_p{p}_{n}_{m}.txt` (soluciones base), `data/haces_p{p}.txt`.

### 5. Cuánto cubrimos (totales A071607×N, conocidos hasta N=31)

Órbitas de traslación totales: 36, 500, 2.288, 243.592, 720.472.148, 13.948.957.032 (p=13…31).
Con estructura: 100 % (13), ~62–74 % (17), 50 % (19), ~3 % (23), <0,01 % (29). El resto crece sin freno.

### 6. Mapa completo de p=17 y p=19 (23 soluciones distintas cada uno)

- **Semillas** (`semillas.py`) [E]: simetrías propias de cada solución (grupo 8(p−1), más traslación).
  p=17: 22 de 23 tienen alguna (1 a 11 reinas semilla); sin ninguna solo la clase 5 ×128.
  p=19: 11 de 23; las 7 con centro (10 semillas) = las RADIAL (504 órbitas); 12 sin ninguna (1.728 órbitas).
- **Intercambios** (`intercambios.py`) [E]: mover k reinas. **Nunca k=3 ni k=5** (p=17 y 19) [O]; mínimo 4.
  p=17, k≤8: 19 de 23 en una componente. p=19, k≤7: componentes de 6 y de 4, 13 sueltas.
  La 2+2 de p=19 = una de dos espirales con 6 reinas cambiadas.
- **Polinomios** (`polinomios.py`): grado casi siempre máximo (p−4); solo son cortos los simétricos. No aporta.
- **Mezclas de dos** (`mezclas.py`) [E]: T = A en unas filas, B en el resto, A y B toroidales
  cualesquiera. p=17: 19 de 23 (azar 0/100); p=19: 14 de 23 (azar 0/30). Tamaños siempre 9+8 o 13+4
  (p=17) y 10+9, 13+6, 15+4, 11+8 (p=19).
- **Por qué esos tamaños** (`mezclas_porque.py`) [E, p=17]: en clases 2, 3 y haces de clase 4 la pieza
  pequeña es UNA recta entera de T y es una clase de μ4 o μ8 alrededor de un centro: mezclar = cambiar
  un bloque = regla de reclamos. En las RADIAL de clase 4: dos clases de μ4 o parejas ± respecto a un
  centro. En la clase 5 ×32: sin centro (órbitas de cierre [1,8,8]). p=19: 6 y 9 dividen a 18 (bajan
  de nivel); 4 y 8 son parejas ± de las que tienen centro [O, deducido de los tamaños].
- **Mezclas de tres** (`mezclas3.py`, `mezclas_niveles.py`): solo valen con piezas sencillas.
  Azar p=17: 0/200 (clase ≤2), 12 % (≤3), 52 % (todas). Azar p=19: 0/40 (≤2), 5 % (≤4), 40 % (≤5).
  Añadir como piezas las soluciones ya descritas más complicadas sube el azar a 32–42 %: no ayuda.
- **Encadenar mezclas de dos no llega a las que faltan**: no son mezcla de dos de NINGÚN par.

Balance (distintas / órbitas de traslación):

| | p=17 (23 / 500) | p=19 (23 / 2.288) |
|---|---|---|
| regla exacta | 19 / 260 | 4 / 56 (clases 1–2) |
| centro de simetría (10 semillas) | – | 7 / 504 |
| receta fiable (mezcla o semillas) | 2 / 48 | 4 / 576 |
| solo indicios | 1 / 64 (clase 6 ×64) | 7 / 1.008 (tres piezas clase ≤4) |
| nada | 1 / 128 (clase 5 ×128) | 1 / 144 |

**Las nueve sin descripción** (`nueve.py`, `fig_nueve.png`): p=17 índice 20; p=19 índices 13–18, 21, 22
(índices = orden (clase, tam) de `data/orbitas_p{p}.json`). Todas sin simetría, 5–6 rectas, no mezcla de
dos. Entre las 8 de p=19 comparten 9,4 reinas de media en la mejor copia frente a 8,0 con el resto
(`parecidos.py`): parecido real pero pequeño; el arranque 0,2,4 común es efecto del representante.

### 7. Conclusión y dos líneas abiertas

Hay un grupo "ordenado" (una espiral, dos, haces n+1, mezclas por bloques, simétricas) que se puede
generar directamente, y un resto sin simetría que crece hasta ser casi todo. Líneas:
(A) completar el grupo ordenado y sus reglas generales (hecho en la sección siguiente: doble reclamo,
generador único, censo de 41 y 43); (B) tratar el resto como tal: contarlo y buscar otra vista o geometría.
Prompts para conversaciones nuevas: `PROMPT_ordenadas.md`, `PROMPT_resto.md`.

## Ordenadas: regla del doble reclamo (haces n+m), generador por bloques y generador único (octubre 2026, 2ª parte)

Scripts nuevos: `haces_regla.py`, `haces_distintas.py`, `bloques_mover.py`, `ordenadas.py`,
`casi_simetria.py`, `intercambio_imposible.py`. `haces.c` cambiado (ver punto 4).
Marcas: **[P]** probado, **[E]** exhaustivo en los p indicados, **[O]** observado.

### 1. Regla del haz 2+2 (y n+m en general): el "doble reclamo" (`haces_regla.py`, `data/haces_regla.txt`)

En un haz n+m cada paralela A_i corta a cada B_j: hay n·m cruces. Desde cada cruce las filas se
reparten en clases de H (clase de r − fila del cruce).

- **Por qué las clases enteras fallan siempre en 2+2** [P, salvo el detalle de las filas de cruce; E: 0 casos]:
  en 2+2 cada recta tiene DOS cruces. Un conjunto de filas que fuese de clases enteras alrededor de dos
  centros distintos quedaría igual al trasladarlo (girar alrededor de un centro y deshacer el giro
  alrededor del otro es una traslación), y en un primo eso obliga a que sea todo o nada.
- **Disjuntas** [P si E(a,b) es grupo, caso (i,−i); E en el resto]: en cada cruce, las dos rectas que se
  cortan ahí no comparten ninguna clase. 2+2: 32/32 (p=19), 160/160 (31), 1.200 (29), 18.240 (37),
  58.432 (41), 1.856/1.856 (43). Azar (mismas rectas y tamaños, filas barajadas): 0.
- **Doble reclamo** [E]: reclamo de una recta = filas que caen en clases suyas vistas desde TODOS sus
  cruces (intersección de clases enteras). Con un solo cruce (haces n+1) es la regla vieja.
  Cada recta ocupa exactamente su reclamo en:

  | haz | p | |H| (clases) | exacta | resto |
  |---|---|---|---|---|
  | 2+2 | 19, 31 | 6 (3, 5) | 32/32, 160/160 | – |
  | 2+2 | 37 | 6 (6) | 672/672 | – |
  | 2+2 | 43 | 6 (7) | 1.664/1.856 (90 %) | solape de 3 filas |
  | 2+2 | 41 | 8 (5), 10 (4) | 160/160, 32/32 | – |
  | 2+2 | 29, 37, 41 | 4 (7, 9, 10) | 1.040/1.200 (87 %), 15.152/17.568 (86 %), 52.176/58.240 (90 %) | solape de 1–7 filas |
  | 3+2 sobre (i,−i) | 29 | 4 (7) | 5.004/6.120 (82 %) | solape de 1–6 filas |
  | 3+1 | 29, 41 (i y |H|=8), 43 | 4, 8, 6 | 6.702/6.702, 380.346/380.346, 7.552/7.552 | – |

  Azar: 0 en todos. Cuando no es exacta, siempre es porque **los reclamos de dos paralelas se solapan**
  (en n+1 no pasa nunca; en n+m aparece cuando hay 7 clases o más) y las filas del solape se reparten
  por la condición entre paralelas. Sigue valiendo "disjuntas".
- **¿Dos espirales + un bloque movido?** [P dado "disjuntas"]: toda 2+2 es dos soluciones de dos
  espirales pegadas (filas de A_i ∪ B_j por un lado, las de la otra pareja por otro), de las dos formas
  de emparejar. [E] Que uno de los trozos sea un bloque (clases enteras alrededor de un centro):
  siempre en p=19 y 31; 656/672 (37, tipo w); 1.632/1.856 (43); 50 % en |H|=8, 10; 66–78 % en tipo i.
  Azar ≈ 0,1 %. O sea: la pista de p=19 es cierta ahí, pero no es la regla general; la regla general
  es el doble reclamo.
- **No es una fórmula cerrada**: describe y comprueba, y reduce la elección a "qué clases reclama cada
  recta en cada cruce", pero contar sigue pidiendo recorrer esas elecciones (como N(p) en 2+1).
- **3+2 y 3+3 de p=19 y 23** [E]: usan pendientes con H = todo (una sola clase). Ahí no hay regla de
  clases que buscar; 5–6 rectas sobre 19–23 filas ya no es estructura.
- **Corrección**: los haces 3+1 sobre pendientes s (p=17: 12 base; p=23: 36; p=41: 156, |H|=10) NO
  cumplen la regla de reclamos. Todos tienen centro de simetría (una recta por el centro y dos
  paralelas simétricas, tamaños x, y, y): son de la familia "centro", no de la de reclamos.
  "Todas las n+1 cumplen clases enteras" vale para las de tipo i, w y |H|=8.

### 2. Mezcla = mover un bloque, como generador (`bloques_mover.py`, `data/bloques_mover_p*.txt`)

Bloque = unión de clases enteras de un subgrupo (el más pequeño: parejas ±) alrededor de una fila
centro, hasta (p−1)/2 filas. Mover = liberar esas filas y recolocar las reinas en las mismas columnas y
diagonales. Azar = lo mismo con filas cualesquiera del mismo tamaño.

| p | bloques que dan otra toroidal | azar | alcanzadas desde las de una espiral |
|---|---|---|---|
| 13 | 4,9 % (4 filas), 2,3 % (6) | 0 %, 0,25 % | 5 de 5 (36/36 órbitas de traslación), 2 rondas |
| 17 | 1,4 % (4), 0,14 % (6), 2,3 % (8) | 0 %, 0,02 %, 0,06 % | 19 de 23 (244/500), 3 rondas |
| 19 | 0,9 % (9 filas), ≈0 el resto | 0 % | 9 de 23 (632/2.288), 6 rondas |

[E] Los bloques de subgrupo grande aciertan mucho más (p=17: |G|=8 14 %, |G|=4 8 %, parejas ± 0,8 %;
p=19: |G|=9 9 %, |G|=6 5 %, parejas ± 0,01 %). En p=17 no se alcanzan 17 (RADIAL), 19, 20, 22.
En p=19 "mezcla de dos = un bloque con centro" solo es cierto en las clases 2 y en la 2+2; las mezclas
de las clases 5–6 (piezas de 4, 6, 8 filas) no son bloques con centro (`mezclas_porque.py 19`).

### 3. Generador único (`ordenadas.py p`, `data/ordenadas_p*.txt|json`)

Familias: F0 una espiral · F1 dos espirales · F2 haces n+m por doble reclamo (parejas con H pequeño y
≥ 3 clases) · F3 centro de simetría (una pendiente por pareja de filas ±r) · F4 cierre por movimientos
de bloque. Cada solución generada se comprueba como toroidal y se compara con el recorrido exhaustivo.

| p | generadas / distintas | órbitas de traslación | no generadas |
|---|---|---|---|
| 13 | 5 / 5 | 36 / 36 (100 %) | – |
| 17 | 20 / 23 | 276 / 500 (55 %) | 19 (c5×32), 20 (c5×128), 22 (c6×64) |
| 19 | 14 / 23 | 992 / 2.288 (43 %) | 13–19, 21, 22 (todas c6×144) |
| 23 | 184 / ? | 19.104 / 243.592 (7,8 %) | – |

[E] Ninguna generada falta en las listas exhaustivas (p=13, 17, 19). p=23, contra el volcado de clase
≤ 5: genera las 20 de clase 1, las 32 de clase 2, **las 396 de clase 4** (incluidas las 264 "de
pendientes corrientes sin regla": salen de F3 y F4) y 1.760 de las 2.552 de clase 5; además 16.896 de
clase ≥ 6. F3 aporta 7.436 y F4 11.616. Tiempo: 1 s, 16 s, 44 s, 2 min.
Honestidad: F0 y F1 son fórmula; F2, F3 y F4 recorren elecciones pequeñas (clases, parejas ±, bloques),
no tableros. Las no generadas de 17 y 19 son las "sin descripción" más 19 y 22 de p=17 y 19 de p=19.

### 4. Censo terminado (`haces.c`, `haces_distintas.py`, `data/haces_distintas.txt`)

`haces.c` ahora rellena en cada paso la fila con menos candidatas (misma búsqueda exhaustiva, mismas
soluciones: p=29 3+1 y 2+2 idénticos byte a byte; p=41 2+2 repite 58.432). El censo entero de un haz
tarda segundos en vez de horas. `PAR="a b" ./haces p n m` limita a una pareja.

| p | 3+1 | 2+2 |
|---|---|---|
| 41 | 2.536.680 = 2.534.840 (i) + 1.040 (s, |H|=10, con centro) + 800 (|H|=8) | 584.320 |
| 43 | 52.864 (todo |H|=6: w, −w y giros) | 19.488 (todo |H|=6) |

Duplicados (soluciones que admiten dos haces; método validado con 64 en p=17 y 88 en p=23): ninguno
en 2+1, 2+2, 3+1 de p = 29, 31, 37, 41, 43 (incluido el 3+1 de p=41: 380.502 base, todas con un
solo haz; 2.536.680 órbitas de traslación distintas).

### 5. La clase 5 ×128 de p=17 no es una simétrica rota (`casi_simetria.py`)

Se prueban las 128·289 simetrías del toro y se cuenta cuántas reinas caen sobre reinas. La ×128:
máximo 8 de 17. Permutaciones al azar: mediana 8 (rango 6–11). Las otras 22 soluciones: 17 de 17.
En p=19 las 12 sin simetría dan 7–10 de 19 (azar 6–12). [E] No hay simetría escondida ni "casi".
Tampoco sale por bloques ni es mezcla de dos: sigue sin descripción.

### 6. Nunca 3 ni 5 reinas: probado (`intercambio_imposible.py`)

Si se recolocan k reinas, cada una hereda la columna de otra (σ), la diagonal / de otra (τ) y la
diagonal \ de otra (ρ), y las cuatro i, σ(i), τ(i), ρ(i) son distintas. [P] Con k=3 no caben cuatro
distintas: imposible para todo p. [P por ordenador] Con k=5 hay 1.344 formas de elegir (σ, τ, ρ); en
todas, las 10 ecuaciones lineales obligan a que dos reinas compartan fila o columna (certificado
entero con D ∈ {1,…,5}): imposible para todo primo p ≥ 7. Con k=4 quedan 6 formas sin obstáculo (las
que se ven). Vale para cualquier colocación sin ataques en el toro, no solo p=17, 19.

### 7. Qué queda abierto

- Fórmula de recuento para 2+2 (hoy: regla + recorrido de reclamos) y por qué el solape de paralelas
  aparece a partir de 7 clases.
- p=17: 19 (c5×32) y 22 (c6×64) tienen simetría propia pero no salen de ninguna familia ni por bloques.
- p=19: la 19 es mezcla 13+6 pero su pieza no es un bloque con centro.
- Las "sin descripción" (p=17: 20; p=19: 13–18, 21, 22) siguen igual: van en `PROMPT_resto.md`.

### 8. "Tres menos una": T = A + B − C (`sumas.py`, `data/sumas_p{17,19}.txt`)

Idea del usuario: sumar dos soluciones sencillas y restar una tercera, fila a fila (módulo p).
A y B entre las de una o dos espirales (todas sus copias), C cualquier toroidal; se prueban los 8 giros
de cada objetivo. Hay que quitar un caso trivial: con A y B de pendientes opuestas sale C = −T, que es
la propia T girada 180°; se descartan las C de la misma órbita que T.

- [E] p=17: salen las 4 de clase 3 y los haces 3+1 / 4+1 (12, 14, 18): de 4.896 a 21.216 formas cada una.
  No salen las de centro de simetría (13, 15, 16, 17, 21) ni las 3 sin explicar (19, 20, 22).
- [E] p=19: no sale ninguna (ni la 2+2). Azar (permutaciones): 0 en los dos primos.
- [P] Por qué: es la regla de reclamos escrita como suma. En un haz 2+1, sea X1 la solución de dos
  espirales "A1 en su reclamo, B en el resto", X2 igual con A2, y C la espiral B. En las filas de A1:
  A1 + B − B; en las de A2: B + A2 − B; en las de B: B + B − B. O sea T = X1 + X2 − B siempre
  (inclusión–exclusión de los reclamos). En p=19 no hay haces n+1, por eso no sale nada.
- Conclusión: la combinación 3−1 es otra forma de ver los haces n+1, no una familia nueva, y no
  alcanza a las sin descripción.

### 9. "Espirales de otro tamaño": p = k1 + k2 − k3 (`tamanos.py`, `data/tamanos_p*.txt`)

Idea del usuario (17 = 11 + 13 − 7). Se escogen k filas de T y se comprimen (filas y columnas
renumeradas en orden). Si queda una solución de k reinas, T "contiene" una de tamaño k. Pregunta: ¿hay
A (k1 filas) y B (k2 filas) que cubren las p filas y cuyo solape de k3 filas también es solución?
Versión toroidal (diagonales módulo k) y normal (tablero plano). [E]

| p = k1 + k2 − k3 | lo tienen (toroidal / normal) | sin explicar |
|---|---|---|
| 17 = 11 + 13 − 7 | solo la espiral de pendiente 2 (11.220 formas); una de dos espirales en la normal (3) | 0 de 3 |
| 17 = 11 + 11 − 5 | solo la espiral de pendiente 2 (14.280) | 0 de 3 |
| 17 = 13 + 13 − 9 | nadie | 0 de 3 |
| 19 = 13 + 11 − 5 | espirales (43.890), la 12 (18) y la 15 (3) | 1 de 9 (la 15) |
| 19 = 13 + 13 − 7 | espiral (50.160); en la normal la 16 (6) | 1 de 9 (la 16, solo normal) |

Permutaciones al azar: 0. La espiral de pendiente 2 lo cumple de miles de formas porque, comprimida,
sigue siendo la espiral de pendiente 2 de cualquier tamaño. En la 15 de p=19 la pieza de 11 sí es la
espiral 0, 2, 4, …, 10, 1, 3, … y la de 5 también, pero la de 13 no es espiral.
Conclusión: sumar soluciones de otros tamaños describe las espirales de pendiente 2 y aparece de
forma suelta en 2 de las 9 sin explicar; no las construye.

### 10. Otras curvas en vez de rectas (`curvas.py`, `data/curvas_p{17,19}.txt`)

Idea del usuario: curvas que pasan una vez por cada fila pero solo tocan algunas reinas. Familias
c = c0 + w(r) módulo p: rectas (w = a·r), parábolas (a·r² + b·r), potencias (a·(r − r0)^k, k = 2…p−2;
incluye la hipérbola a/(r − r0) y las espirales de la rejilla de potencias), exponencial (a·g^r).
Para cada solución: mínimo de curvas que tocan todas las reinas (exacto, MILP) y la curva con más
reinas. Azar: 30 permutaciones al azar. [E]

| p=17, mínimo de curvas | rectas | parábolas | potencias | exponencial |
|---|---|---|---|---|
| sin explicar (19, 20, 22) | 5, 5, 6 | 4, 4, 4 | 3, 3, 3 | 4, 4, 4 |
| azar | 5–6 (4 en 1/30) | 4 (30/30) | 3 (25/30), 4 | 4 (26/30), 5 |

| p=19, mínimo de curvas | rectas | parábolas | potencias | exponencial |
|---|---|---|---|---|
| sin explicar (9) | 6 (9/9) | 4 (4), 5 (5) | 4 (9/9) | 4 (2), 5 (7) |
| azar | 6 (29/30) | 4 (13), 5 (17) | 4 (29/30) | 4 (2), 5 (28) |

La curva con más reinas tampoco se distingue (p=19 potencias: 7–8 sin explicar, 7–9 azar).
Conclusión: con ninguna de estas geometrías las sin explicar se parecen a nada; se comportan como una
permutación al azar. De paso: las curvas de potencias sí "entienden" parte de lo explicado: en p=17
las de centro de simetría 12 y 13 necesitan 2 curvas (4 rectas) y una de dos espirales (la 3) es UNA
sola curva de potencias (es la rejilla de potencias). Sin probar: curvas en un tablero más grande que
el de p (incrustar la solución en N mayor).

### 11. Hélices de paso cualquiera en el cilindro (`helices.py`, `data/helices_p{17,19}.txt`)

Idea del usuario: en el cilindro (fila = altura, columna = ángulo) una curva que recorre todas las
alturas y solo corta algunas reinas.
- [P] Si la curva tiene que pasar EXACTAMENTE por casillas y es una hélice de paso fraccionario (o un
  polinomio de coeficientes fraccionarios), solo cae en casillas cada q filas y esas casillas están
  sobre una recta módulo p: son trozos de las espirales de siempre, nada nuevo.
- Lo nuevo: hélice de paso real s que atraviesa la casilla floor(s·r + c0) en cada fila ("recta
  digital", palabras sturmianas). Se generan todas las distintas sobre p filas (8.449 en p=17, 13.205
  en p=19) y se mide el mínimo de hélices que tocan todas las reinas. [E]

| | p=17 mínimo | mejor hélice | p=19 mínimo | mejor hélice |
|---|---|---|---|---|
| sin explicar | 4 (3/3) | 7 | 4 (9/9) | 6–9 (casi siempre 7) |
| azar | 4 (30/30) | 6–9 | 4 (30/30) | 6–8 |
| explicadas | 1–4 | hasta 17 | 1, 2 o 4 | hasta 19 |

Conclusión: también con hélices de cualquier paso las sin explicar se comportan como una permutación
al azar. Con las cinco geometrías probadas (rectas, parábolas, potencias, exponenciales, hélices
digitales) el resultado es el mismo.

### 12. Combinaciones lineales de hélices y de soluciones de 13 en el cilindro de 17 (`combina.py`)

Idea del usuario: sumar curvas que cortan en puntos distintos. Piezas: las 8.449 hélices de paso real
y las 4.524 toroidales de 13 metidas en el cilindro de 17 (ángulo estirado 13→17, filas repetidas o
estiradas: 9.048 piezas distintas). Combinaciones: T = λ·X + μ·Y (λ, μ cualesquiera módulo 17) y
T = X + Y − Z con hélices. [E, p=17]
- Solo salen las de una espiral (trivial: suma de rectas es una recta). Ninguna otra de las 23: ni las
  de dos espirales, ni las de haz, ni las sin explicar. Azar: 0/20.
- Por qué: sumar funciones mezcla las columnas de todas las filas a la vez y rompe "una reina por
  columna y por diagonal". Lo que funciona en lo explicado es ELEGIR, fila a fila, qué curva manda
  (dos espirales, haces, mezclas), no sumar. Con signo ± la suma de k hélices se queda a ≤ k casillas
  de una sola hélice [P], así que no puede dar nada lejos de una recta.
