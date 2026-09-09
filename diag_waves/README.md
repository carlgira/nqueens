# diag_waves — N-reinas como ondas

Las permutaciones ya resuelven filas y columnas. Todo el problema vive en los dos
canales diagonales `d⁺(i) = i + σ(i)` y `d⁻(i) = i − σ(i)`. Este directorio
explora qué pasa al codificarlos como ondas, hélices y superficies.

**Resumen en una frase**: la representación ondulatoria *describe* el problema con
precisión extraordinaria y *no lo construye*. Todo lo cuadrático colapsa a un único
testigo exacto; lo que no es cuadrático es ruido; y la información que distingue
una solución de otra se destruye en cuanto sumas.

---

## La idea base: la diagonal va en la FRECUENCIA

La intuición inicial era poner el número de diagonal como **amplitud**
(`d₁·cos(x) + d₂·cos(2x) + …`). Eso es una DCT invertible: no pierde nada pero
tampoco expone la restricción.

Poniéndola en la **frecuencia** se cuocienta exactamente la simetría que no importa
(qué fila lleva qué diagonal) y queda el invariante que sí:

```
W(x) = Σᵢ exp(i·d(i)·x)        ⟨|W|²⟩ = Σᵥ cᵥ² = N + 2·(pares en conflicto)
```

En forma de serie real, con amplitud = ocupación de la diagonal y frecuencia = el
índice de la diagonal:

```
R(x) = Σₖ cₖ·cos(kx)           conflictos = ⟨R²⟩ − N/2
```

**Válido ⟺ todas las amplitudes son 0 ó 1.** Nunca un `2·cos(kx)`.

→ `diag_waves.py`

---

## Frente 1 · Recortar el espacio de ESPECTROS

**Objeto**: una hélice completa. **Pregunta**: ¿existe un tablero que la produzca?

### Los invariantes

Los tres primeros momentos son **constantes para toda permutación**:

```
M₀ = Σ cₖ     = 2N
M₁ = Σ k·cₖ   = 4N²
M₂ = Σ k²·cₖ  = ⅔N(N−1)(2N−1) + 2N(N−1) + N + 9N³
```

Geométricamente son las derivadas de la hélice en el origen: `Z(0)=M₀`,
`Z'(0)=i·M₁`, `Z''(0)=−M₂`. **Posición, velocidad y curvatura en el origen son
idénticas para todas.** Por eso todas las curvas salen juntas y se separan solo a
partir de la tercera derivada.

Contar espectros compatibles se hace por DP, sin enumerar nada:

| N | pares posibles | + M₁ | **+ M₂** | realizables |
|---|---|---|---|---|
| 12 | 1.8·10¹² | 1.06·10⁹ | **13 153** | 7 605 |
| 14 | 4.0·10¹⁴ | 1.46·10¹¹ | **271 852** | 171 240 |
| 16 | 9.0·10¹⁶ | 2.2·10¹³ | **7 860 935** | 5 363 379 |

→ `bounds.py`

### Cotas de envolvente

Enumerando los factibles se obtienen techo y piso de `S(x)` calculables **solo con
N**. Exactos de N=4 a N=9 (ej. N=9: techo +6.640, piso −7.875, idénticos al real).

### El único invariante universal en racionales de π

Búsqueda exhaustiva sobre `m = 2..18`, todo `j` coprimo:

```
N par   ⟹  S(π) = 0   para TODA permutación
N impar ⟹  S(π) = 2(N−2e),  e ≡ N (mod 2)  →  retícula de paso 8
```

Los **fantasmas no son permutaciones**, así que pueden violarlo. Es el único filtro
nuevo, exacto y barato contra ellos: **caza el 9% (N=10), 13% (N=12), 23% (N=14)
sin perder ni un real**, en O(N).

### El discriminador que sí funciona

Realizables vs fantasmas dentro del conjunto factible — todos con la misma energía,
área y momentos:

| entrada | AUC prueba |
|---|---|
| `cₖ` crudo | 0.700 |
| hélice Re+Im | 0.710 |
| **hélice `\|Z\|` (sin fase)** | **0.992** |
| autocorrelación de `c` | 0.999 |

`|Z|²` es la transformada de la **autocorrelación** de `c`. Ningún resumen escalar
de ella sirve (max, varianza, energía aditiva: 0.50–0.57) y la sonda lineal da
0.60: **es un patrón sobre ~8–12 desfases, no un número**.

Aflojando la exactitud un 1%, elimina el **90%** de los fantasmas.

→ `discriminator.py`, `multitask.py`

### Lo que NO distingue fantasmas

Todo lo **cuadrático** es idéntico por construcción:

| cantidad | reales | fantasmas |
|---|---|---|
| radio RMS | 4.899 | 4.899 |
| área | 1809.557 | 1809.557 |
| volumen `2π²(2N+2c)` | 473.74 | 473.74 |

Y todo lo que lleva **raíz cuadrada** es ruido: perímetro, longitud de arco,
superficie del sólido de revolución.

### Cómo son los fantasmas

- Se bloquean tarde: media **8.25 de 12 filas**, el 53% en la fila 9, ninguno pasa
  de 10.
- A **distancia mínima** de los reales (6 coordenadas = 3 intercambios).
- Cerrados bajo las **mismas** simetrías (espejo e inversión, 100%).
- Perfil de ocupación casi idéntico (diferencia máxima 0.034).

---

## Frente 2 · Recortar el ÁRBOL de búsqueda

**Objeto**: un tablero a medio llenar. **Pregunta**: ¿esta rama lleva a solución?

El 80–94% de los estados parciales a profundidad media están **condenados**.

### La fórmula

```
score(condenado) = σ(
   −4.411 · (grado medio        − 0.1281)/0.0960
   −2.801 · (profundidad d/N    − 0.6965)/0.1472
   +1.406 · (casillas libres    − 0.0518)/0.0516
   −0.797 · (fila más pobre     − 0.0633)/0.0680
   −0.296 · (concentración d⁻   − 0.3041)/0.3143
   +0.296 · (concentración cols − 0.3548)/0.3131
   −0.261 · (concentración d⁺   − 0.3041)/0.3143
   +2.737 )
```

Los siete rasgos son **adimensionales**, y por eso transfiere entre N.

| modelo | AUC |
|---|---|
| red (127 rasgos) | 0.951 |
| lineal (127 rasgos) | 0.925 |
| **la fórmula (7 rasgos)** | **0.905** |

### Poda real

| umbral | nodos (N=12) | ahorro | soluciones conservadas |
|---|---|---|---|
| — | 856 189 | — | 100% |
| 0.999 | 443 208 | **48%** | **86%** |
| 0.5 | 117 012 | 86% | 10% |

Falsos positivos por profundidad: 1.6%, 3.0%, 6.5%, 5.5%, **0%**, **0%**. La
pérdida del 14% es **acumulación** entre niveles, no imprecisión. Podando solo en
profundidad ≥10, la pérdida es cero.

### Transferencia

Entrenado **una vez** con N=10,11,12 (1 segundo de datos):

| N | 13 | 20 | 32 | 50 | 80 | **100** |
|---|---|---|---|---|---|---|
| AUC red | 0.937 | 0.878 | 0.941 | 0.956 | 0.949 | **0.955** |

No se degrada en un factor 8 de N. *Aviso*: en N=100 el 45% de los estados
muestreados superó el cap de 600k nodos al etiquetar y se descartó; ese sesgo podría
inflar la cifra.

**Los dos frentes no se cruzan**: meter esta fórmula dentro del emparejamiento del
frente 1 poda el 87% y mata el 100% de las soluciones. Las distribuciones son
distintas y el rasgo dominante se satura.

→ `prune.py`, `_big.py`

---

## Frente 3 · El espectro PENDIENTE  (lo que sí poda)

Los frentes 1 y 2 miran **lo colocado**. Este mira **lo que falta**, y es la única
poda exacta con ahorro real de toda la sesión.

### La condición

A las `N−r` reinas pendientes les hacen falta `N−r` diagonales `d⁺` distintas,
`N−r` diagonales `d⁻` distintas y `N−r` columnas distintas. Si de alguna de las
tres hay menos disponibles, el estado está **muerto con certeza**.

Es la **condición de Hall**, y comprobarla con un emparejamiento bipartito
equivale a comprobar los `2^(N−r)` subconjuntos, en `O(V·E)`.

### Cuánto caza (N=11, 147 150 estados condenados)

| versión | cazados | vivos perdidos |
|---|---|---|
| conjunto completo (1 subconjunto) | 58.3% | **0** |
| sub-bloques (~30 subconjuntos) | 70.9% | **0** |
| **Hall completo (emparejamiento)** | **77.0%** | **0** |

Por profundidad, con Hall completo: 4.0% (prof 5), 28.8% (6), **72.4% (7)**,
92.8% (8), 98.4% (9), **100% (10)**. Un segundo para los 147k estados.

### Por qué no llega al 100%

Hay **tres** restricciones simultáneas. Satisfacerlas a la vez es emparejamiento
**3-dimensional**, NP-duro. Se comprueba cada una por separado —cada bipartita es
exacta y necesaria— pero las tres juntas no bastan. Ese 23% que escapa es la
dificultad irreducible: si las tres bipartitas bastaran, N-reinas sería polinómico.

### En el árbol libre

| N | nodos DFS | nodos con poda | ahorro | tiempo |
|---|---|---|---|---|
| 11 | 166 926 | 109 414 | 34.5% | 4.9× más lento |
| 12 | 856 189 | 548 489 | 35.9% | 4.9× |
| 13 | 4 674 890 | 2 924 918 | 37.4% | 4.7× |
| 14 | 27 358 553 | 16 518 309 | **39.6%** | 4.9× |

El ahorro **sube** con N (+2 puntos por unidad) pero la relación de tiempo se queda
**plana en 4.9×**. No se cruzan en ningún N alcanzable.

Comparado con el forward-check clásico (fila sin opciones): 15.4% menos de nodos.

### En el EMPAREJAMIENTO, muerde mucho más

Ahí las diagonales están restringidas al espectro, así que quedarse sin opciones es
mucho más probable:

| N | nodos normal | con poda | **ahorro** | relación de tiempo |
|---|---|---|---|---|
| 10 | 21 557 | 10 581 | 50.9% | 3.2× |
| 11 | 69 472 | 28 366 | 59.2% | 3.2× |
| 12 | 113 923 | 39 961 | 64.9% | 3.0× |
| 13 | 196 988 | 59 716 | **69.7%** | **2.8×** |

Aquí **sí hay tendencia**: ahorro +6 puntos por N y relación bajando. Las dos curvas
convergen; alrededor de N≈18–20 debería cruzar el equilibrio incluso en Python.

Y acelera justo el cuello del frente 1 — en el embudo de N=14, el emparejamiento
era 88 de los 89 segundos.

→ `lookahead.py`, `hall.py`

### Lo que NO funciona en este frente

- **`S²` en estados parciales**: el backtracking nunca coloca una reina en
  conflicto, así que `Σcₖ² = 2r` **siempre**. Un solo valor por profundidad, cero
  información.
- **Momento parcial `M₂`**: sí varía, pero el rango de los condenados **contiene**
  al de los vivos en toda profundidad, y en prof 9 son idénticos. No hay lista
  blanca.
- **Representación más gruesa** (diagonales módulo `m`): apiña más soluciones pero
  **rompe la exactitud** — dos diagonales distintas pueden compartir residuo, así
  que la condición de conteo deja de ser necesaria. Lo que sí funciona es lo
  contrario: mirar en **sub-bloques**, más fino.

### El embudo geométrico, y por qué sobra

Aplicando a todas las permutaciones filtros de forma de la curva (techo, piso,
`⟨S⁴⟩`) antes del testigo:

| N | permutaciones | + techo | + piso | + `⟨S⁴⟩` | + testigo |
|---|---|---|---|---|---|
| 9 | 362 880 | 37.4% | 29.0% | 5.6% | **0.097%** |
| 10 | 3 628 800 | 79.8% | 57.7% | 1.7% | **0.020%** |

Pero **`⟨S²⟩ = N` implica las otras tres**: de las que pasan el testigo, el 100%
pasa también techo, piso y `⟨S⁴⟩` (92/92, 352/352, 724/724). Y el testigo es el más
barato: `O(N)` contra `O(N·G)`, 3× y 36× más baratos respectivamente.

Un pre-filtro solo compensa si es más barato que lo que evita. Aquí es al revés.
El embudo mide **cuánta información hay en la forma** (÷34 con `⟨S⁴⟩`), no es un
algoritmo. → `funnel.py`


---

## La pared, dicha de seis formas

El espectro registra **qué diagonales** están ocupadas, nunca **quién con quién**.

1. **Colisiones**: 92 soluciones de N=8 → 58 hélices; 724 de N=10 → 434. Dos
   tableros distintos, onda idéntica coeficiente a coeficiente.
2. **No son simetrías**: solo el espejo de columnas actúa limpiamente sobre el
   espectro (100%); transpuesta 34%, rot 90° 0%. Detectar simetría desde la hélice:
   sensibilidad 22%, precisión 45%.
3. **Fantasmas**: 328 de 762 espectros en N=10 pasan momentos, energía, área y
   `⟨S²⟩ = 2N`, y ninguna permutación los realiza.
4. **Decodificar falla**: hélice → tablero da 0% exacto, incluso entrenando solo con
   hélices **únicas** (sin ambigüedad): 100% en entrenamiento, 0% en prueba.
5. **La disyuntiva es estructural**: poniendo el par `(d⁺,d⁻)` en la frecuencia la
   codificación se vuelve inyectiva **y pierde el testigo** (`⟨|W|²⟩ = N` siempre),
   porque `d⁺+d⁻ = 2i` impide las colisiones. Detectar conflictos exige que las
   cosas choquen; identificar exige que nunca choquen.
6. **En 2D no hay disyuntiva**: `W(u,v) = Σᵢ exp(2πi(iu+σ(i)v)/M)` es inyectiva
   (5040 distintas para 5040 permutaciones en N=7), se decodifica con una DFT
   inversa, y sus cortes `u=±v` son los dos testigos. El plano identifica, las
   rectas detectan.

→ `wave2d.py`, `paired.py`, `decoder.py`

---

## Resultados sueltos que sí quedaron

**Muestreo**: la hélice necesita `4N` muestras para recuperar `cₖ` exacto; su sombra
real necesita `8N`. **Factor 2 exacto** — Nyquist cobra por tirar la fase. Y el test
`⟨|Z|²⟩ = 2N` con `M` puntos nunca da falsos positivos, pero necesita **~3.7N**
puntos para aceptar todas las soluciones (no `N`). → `sampling.py`

**El perfil universal**: la ocupación media de cada diagonal, sobre todas las
soluciones, colapsa a una curva **independiente de N**:

```
⟨cₖ⟩ = 1 / (1 + exp(5.242·(u² − 0.302)))        u = (k−N)/N
```

Es la distribución de máxima entropía compatible con `M₀`, `M₁`, `M₂`. Su entropía
estima `log₂ Q(N)` con menos de 1 bit de error en N=15–16.
→ `profile.py`, `ensemble.py`

**Las cuencas**: con el testigo como potencial e intercambios como movimiento, toda
solución tiene cuenca, casi todas del mismo tamaño (Gini 0.147), de profundidad
**≈ N/4** (constante de N=8 a N=50). **Teselan**: a distancia 3, el 95%+ cae en
*alguna*. Pero son **esponjas porosas**: solo el 17% de los 28 vecinos sigue dentro,
y el 44% de los puntos pertenece a 2–10 cuencas según el desempate. El **centroide**
de una cuenca recupera la solución exacta (masa 77–89% sobre sus casillas).
→ `basins.py`, `cones.py`, `shape.py`, `fast_basins.py`

**La rotación Sturmiana**: `xᵢ = (αi+β) mod 1`, entropía **cero**, es el mejor
generador encontrado — 33% de aciertos frente al 0.003% del azar en N=13. Los mapas
caóticos aciertan **al nivel del azar o por debajo**; barriendo el mapa estándar, la
tasa cae monótonamente con el caos (33% → 0.015%). La familia satura en G=600: solo
**598 permutaciones para N=13**, de 6.2·10⁹. → `sturmian.py`

**Un representante por órbita**: el convenio (menor lexicográfico) los amontona
(47–66% de huecos diminutos). Eligiendo por barrido queda **mejor que aleatorio**
(sd 0.60 vs 1.0 en N=9). → `orbits.py`

---

## Lo que se descartó, con su número

| idea | resultado |
|---|---|
| independencia de las dos diagonales | `p_ambos/(p⁺p⁻)` decae; precisión 91%→57% (N=7→10), mínimo en N=11 y luego **sube** a 68% en N=16 |
| suma plana `R⁺+R⁻` | no separa; hacen falta bandas disjuntas |
| índice diagonal conjunto | siempre inyectivo, no detecta nada |
| área bajo la curva | `⟨S⟩ = 0` idénticamente |
| segundo momento por banda | rangos idénticos, 0% |
| puentes modulares conjuntos | ya implícitos en los momentos, poda 4–6% |
| clustering / PCA / variedad 1D | silueta 0.08 vs nulo 0.07 (control positivo 0.29); dimensión de correlación 17; sin simetría cíclica |
| curvas polares clásicas | 10 400 limaçons → **17** conjuntos distintos; aciertan 1 de 180 |
| flor de K armónicos | necesita K=7 de 8 para representar; con K=3 sesga 14× y ahí se acaba |
| VAE generativo | 165× el azar pero **colapso de modos** (50 soluciones de 14 200); pierde 7× contra la Sturmiana |
| GA de péndulos | resuelve hasta N=16; el R² de progresión aritmética estaba **en la mediana del nulo** |
| poda del árbol con invariantes exactos | momentos, paridad, área, energía aditiva: **1.00×** en los cuatro |

---

## Trampa metodológica recurrente

Apareció **cuatro veces** y merece regla propia:

> Un test de **umbral** (`≤ máximo`) siempre es legítimo.
> Un test de **pertenencia a conjunto** solo vale si el estadístico toma pocos
> valores. Si el número de valores distintos se acerca al número de candidatos, no
> estás filtrando: estás consultando la respuesta.

Casos: `⟨S⁴⟩` sobre el cuerpo (762 valores para 762 espectros), perfiles modulares
con m≥5 (inyectivos), R² de progresión aritmética (nulo en 0.95), y el filtro de
perfiles completos que "alcanzaba el techo".

---

## Anexo · El mismo experimento en ARRAYS DE COSTAS

Al agotar N-reinas se repitió todo sobre un problema de la misma familia, para
separar qué es del problema y qué de la representación.

### El problema

Una permutación donde **todos los vectores de desplazamiento entre pares de puntos
son distintos**. N-reinas prohíbe 2 direcciones (`d⁺`, `d⁻`); Costas las prohíbe
todas.

Verificado: los arrays de Costas son **exactamente** nuestro testigo aplicado a
todos los desfases a la vez (coincidencia con OEIS A008404 en N=3..9).

### Por qué es mejor terreno, sobre el papel

- **La pregunta abierta es de existencia**, no de conteo: no se conoce ningún array
  de orden 32 ni 33, y se desconoce si existen.
- **El espacio es diminuto**: 56 arrays en N=26, frente a los 10¹⁷ de N-reinas.
- La enumeración exhaustiva llega a **N=28** y costó ~70 años de CPU.
- La secuencia **sube hasta N=16 (21 104) y se desploma** — nada que ver con el
  crecimiento explosivo de N-reinas.
- La onda 2D es **inyectiva**, así que aquí no hay fantasmas.

### El testigo, subido un orden

```
N-reinas:  onda 1D del espectro diagonal   ⟨|W|²⟩ = 2N + 2·conflictos
Costas  :  onda 2D del tablero             ⟨|W|⁴⟩ = 2N² − N + 2·conflictos
```

Verificado a `1e-11` sobre todas las permutaciones de N=5..8. Coste: una FFT 2D,
22–52 µs para N=8..24.

La razón es limpia: `|W(u,v)|²` es la transformada de la autocorrelación 2D
`A(dx,dy)`, y la condición de Costas es exactamente `A ≤ 1` fuera del origen.

### El perfil de energía es una recta

Con una hélice por desfase, `Z_d(x) = Σᵢ exp(i·(p[i+d]−p[i])·x)`:

```
⟨|Z_d|²⟩ = (N−d) + 2·conflictos_d

Costas  ⟺  el perfil es EXACTAMENTE la recta N−d
```

Los 444 Costas de N=8 dan `[7,6,5,4,3,2,1]` clavado, error `7×10⁻¹⁵`. Exceso total:
0 exacto en Costas, ≥ 2 en el resto. → `costas_helix.py`

Aquí el testigo **es un perfil de N−1 componentes**, no un escalar. Es la única
diferencia estructural real entre los dos problemas.

### Invariantes con fórmula cerrada

Constantes para toda permutación:

```
Σ A(dx,dy)        = N(N−1)
Σ dx·A = Σ dy·A   = 0
Σ dx²·A = Σ dy²·A = N²(N²−1)/6
Σ |dx|·A          = N(N²−1)/3
```

Lo que varía es `Σ dx·dy·A`, con rango más estrecho en los Costas
(`[−70,+70]` vs `[−100,+100]` en N=5) — condición necesaria que no separa sola.

### El árbol de búsqueda

| N | nodos | Costas | nodos por Costas |
|---|---|---|---|
| 10 | 304 201 | 2 160 | 141 |
| 12 | 6 977 361 | 7 852 | **889** |

El coste por solución se multiplica por **2.7 cada N**. Y las ramas mueren antes:
**95–99% de condenados desde profundidad 6** (en N-reinas era 80–94%).

### La poda, y por qué transfiere mal

| condición | caza | dónde |
|---|---|---|
| columna sin opciones | 54.6% | solo prof ≥ 6 |
| + conteo de filas | 61.0% | solo prof ≥ 6 |
| + Hall (emparejamiento) | **62.9%** | solo prof ≥ 6 |
| **conteo por desfase** | **0%** | nunca |
| **conteo total de vectores** | **0%** | nunca |

A profundidad 5 hay 31 824 condenados y **no caza ninguno**.

La razón se calcula: para el desfase `d` hacen falta `N−d` diferencias de un rango
de `2N−1` — **sobra la mitad**. Globalmente, 55 vectores de 210 posibles. Nunca
escasean. En N-reinas la presión era real (`N−r` diagonales de `2N−1`) y por eso el
conteo mordía.

**En Costas la muerte de una rama no viene de agotar recursos sino de colisiones
específicas**, y eso ninguna condición de conteo lo ve.

### La geometría, medida otra vez

| cantidad | Costas | resto | veredicto |
|---|---|---|---|
| `∫\|W\|²` (volumen) | 7 = N | 7 = N | constante |
| pico `max\|W\|²` | 49 = N² | 49 = N² | constante |
| **`∫\|W\|⁴`** | **91 exacto** | **≥ 95** | **el único que separa** |
| `∫\|W\|` | [2.30, 2.37] | [1.90, 2.34] | solapa |
| superficie | [6.13, 6.45] | [5.70, 9.50] | solapa |
| rugosidad | [7.79, 7.98] | [7.89, 12.65] | solapa |
| entropía, planitud, nº de máximos, nº de ceros | — | — | solapan |

Idéntico al patrón de N-reinas: **lo cuadrático colapsa al testigo, lo que lleva
raíz es ruido**.

### Los momentos superiores: un falso positivo corregido

`∫|W|⁶` y `∫|W|⁸` **parecían** separar en N≤7. Con el conjunto **completo** de
no-Costas (no una muestra de 400):

| N | Costas máx | no-Costas mín | separa |
|---|---|---|---|
| 7 | 1 813 | 1 855 | sí |
| **8** | **2 912** | **2 870** | **NO** |

Era un artefacto de N pequeño más submuestreo. Tampoco tienen fórmula (ningún
cociente con `N⁴`, `(2N²−N)²` o `N²(N²−1)/2` sale constante) ni distinguen las
construcciones algebraicas (Welch en N=6 da `m6 ∈ [1008,1068]`, contenido dentro
del `[996,1068]` del resto).

→ `costas.py`, `costas_search.py`, `costas_helix.py`

---

## Conclusión sobre los dos problemas

```
restricción de orden 2 (N-reinas) → testigo ⟨|W|²⟩
restricción de orden 4 (Costas)   → testigo ⟨|W|⁴⟩
```

En ambos casos la transformada espectral da **exactamente un invariante**, el del
orden de la restricción, y nada más. Todo lo cuadrático colapsa a él; todo lo que
lleva raíz cuadrada es ruido; los momentos superiores prometen y no cumplen.

No es que no encontráramos la propiedad emergente: está **medido** que en esta
traducción no la hay, y con razón estructural — Fourier convierte una restricción de
orden `k` en un momento de orden `k`, y ahí se acaba el diccionario.

**Lo que quedó sin probar** es la otra familia: relajaciones **LP/SDP tipo
Delsarte**, donde las sumas de ondas no describen el objeto sino que **acotan
cuántos puede haber**. Lo rozamos —el conteo por DP de espectros factibles es el
vértice de un politopo definido por los momentos— pero nunca lo planteamos como
programa lineal. Es el único camino de esta familia que ataca el problema por donde
históricamente sí ha cedido.


---

## Ejecutar

```bash
venv/bin/python -m diag_waves.diag_waves      # el testigo y el multiplexado
venv/bin/python -m diag_waves.bounds          # momentos, cotas, invariantes
venv/bin/python -m diag_waves.discriminator   # realizable vs fantasma
venv/bin/python -m diag_waves.prune           # poda del arbol
venv/bin/python -m diag_waves.helix           # la helice
venv/bin/python -m diag_waves.cones           # las cuencas
venv/bin/python -m diag_waves.sturmian        # la familia de entropia cero
```

```bash
venv/bin/python -m diag_waves.costas_helix     # el perfil N-d de Costas
```

Figuras `.png` en este mismo directorio.


---

# Sesión 2: el embudo, la simetría y los niveles

Segunda tanda. La pregunta cambió: en vez de *describir* las soluciones, **recortar
el espacio** y, al final, **contarlas**. Aquí está lo que sobrevivió.

## El embudo · `pp.py`

Objeto: **todas** las permutaciones de N. Se aplica una regla, se cuenta lo que
queda, se comprueba que no se pierde ninguna solución. Toda regla se deriva **solo
de N**: la cantera son los espectros que cumplen los tres momentos, calculables sin
resolver nada.

| | qué mide | N=10 deja |
|---|---|---|
| — | 3 628 800 permutaciones | 100% |
| **R2** | envolvente puntual, bandas atadas por M₂ | **0.68%** |
| **R3** | momento M₃ (la 3ª derivada) | 0.53% |
| **R4** | momento M₄ | 0.52% |
| **R5** | el invariante `{\|W⁺\|,\|W⁻\|}` | **0.18%** |

**R2 es casi todo.** De los tres momentos, los dos que actúan *dentro* de cada banda
no sirven; el que las **ata** deja el espacio en menos del 1%. En N=7 y N=8 R2 sola
deja exactamente las soluciones.

Y R2 se come a dos reglas que parecían prometer:

- **`S(π)`** — para N par vale 0 para *toda* permutación (`S(π) = (2E−N)((−1)^N−1)`),
  así que no poda nada. Para N impar corta el 20%, pero R2 ya lo hace.
- **La energía de la cola** `⟨S²⟩` en `x>0.5` — poda mucho, pero `corr` con el número
  de conflictos = **0.98**. Es el testigo con ruido: una versión *peor* del exacto.

### El derrumbe

Por muestreo, hasta N=14:

| N | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|
| R2 deja | 0.79% | 0.22% | 0.19% | 0.67% | 5.1% | 6.0% | 18.4% | **27.9%** |

Y la cota **es exacta**: coincide dígito a dígito con la envolvente real de las
soluciones en N=5,6,8,9,12,13. Ese es el resultado incómodo — **exacta y a la vez
inútil**: saber el máximo y el mínimo de `S(x)` no dice qué curvas son soluciones.
La causa es que el conjunto factible (22 espectros en N=7, 271 852 en N=14) crece
mucho más rápido que las soluciones, y el máximo se toma sobre todos ellos.

### Los picos

`S(x)` es un polinomio trigonométrico de grado `4N−1`. Una solución gasta 2N
armónicos distintos con amplitud 1 — la configuración más rizada posible — y cada
conflicto funde dos armónicos y alisa la curva. Es **monótono**:

```
conflictos    0      1      2      3      4      5      6      7      8
picos      30.87  30.46  30.19  30.07  29.95  29.86  29.79  29.71  29.61
```

Pero la caída total es de **1.26 picos** sobre un rango de 25 a 37. Sola poda el 1%;
detrás del embudo, 0.4%. El contador de picos está determinado por el grado efectivo,
que es función del número de conflictos: el testigo mal medido, otra vez.

→ `pp.py`, `fig_pp_all.png`, `fig_pp_peaks_n10.png`, `fig_pp_decay.png`

## La simetría · `orbit_geo.py`

**Las 8 soluciones de una órbita no se distinguen en el módulo.** Girando 90°:

```
d⁺ → (N−1) − d⁻          d⁻ → d⁺ − (N−1)
```

intercambia los canales con reflexión y traslación. Y ninguna de las tres operaciones
(reflexión, traslación, negación) toca `|W|`. Luego el par no ordenado
`{|W⁺(x)|, |W⁻(x)|}` es un **invariante exacto del grupo diedrico**. Verificado en
N=7..10 sin excepción.

Lo que sí varía dentro de la órbita —`S(x)`, `|Z|`, M₃, M₄, hasta 8 valores— varía
por **cómo montamos las dos bandas en una señal**, no por geometría. Ese montaje
rompe la simetría a mano.

### Las colisiones

El invariante no separa órbitas distintas. En N=10, 92 órbitas dan 60 curvas. Y
**todas** las colisiones son el mismo fenómeno:

```
N= 8   2 pares que chocan   con los dos conjuntos de diagonales IDÉNTICOS:  2
N= 9  13 pares                                                            13
N=10  40 pares                                                            38
```

Dos tableros que ocupan **exactamente las mismas diagonales**, y solo difieren en
cómo se emparejan. Comparten `c`, `S(x)`, `|Z|`, M₃, M₄ — todo. En el plano
`(d⁺, d⁻)`: **mismos márgenes, distinta nube**. Es la pared de toda la investigación,
aislada en su forma mínima y con ejemplos explícitos.

La jerarquía completa de N=9: `352 soluciones → 180 ondas S(x) → 46 órbitas → 35
curvas invariantes`. El invariante hace exactamente el cociente correcto (colapsa D4
y nada más); lo que le sobra son 11 órbitas que ninguna representación espectral
puede distinguir.

### ¿Forman un patrón?

El invariante mínimo es el par de autocorrelaciones: `2N−2` enteros por canal. Si las
órbitas formaran una secuencia, vivirían en dimensión 1 o 2 y esa dimensión no
crecería con N. Medido:

| N | 8 | 9 | 10 | 11 |
|---|---|---|---|---|
| órbitas | 12 | 46 | 92 | 341 |
| dimensión para el 90% de varianza | 6 | 11 | 15 | **20** |

Crece como **~2N**. No hay patrón, ni curva, ni sucesión.

→ `orbit_geo.py`, `fig_orbit_inside_n8.png`, `fig_orbit_pair_n9.png`, `fig_orbit_dim.png`

## Los cuellos de botella · `necks.py`, `paths.py`

En `x = 2πj/m` la onda colapsa a un conteo modular y solo cae en una retícula finita
de valores — los **cuellos**.

**Cuándo concentra la simetría**, con condición exacta:

```
S(x) es invariante del grupo diedrico  ⟺  2N·x ≡ 0 (mod 2π)  ⟺  m | 2N
```

N=7 (2N=14) → m ∈ {2,7,14}; N=8 → m ∈ {2,4,8}. Fuera de ahí la órbita se reparte.
Consecuencia: en los puntos que concentran, «quedarse con una de la órbita» no da nada.

### Lo que sí salió: las soluciones ocupan la mitad de los cuellos

```
N=12, x=2π/5:   14 cuellos usados por las 14 200 soluciones
                30 cuellos usados por 14 200 permutaciones al azar
```

Contra un control de 25 grupos al azar del mismo tamaño: **p < 0.001 en N = 9..12**,
y la separación *crece* con N. Es la primera propiedad de las soluciones de todo el
proyecto que es estadísticamente significativa contra un control.

### Y el predictor derivable

```
corr( nº de espectros FACTIBLES en un cuello , nº de soluciones en ese cuello )
   N=9: 0.994    N=10: 0.996    N=11: 0.999    N=12: 0.997
```

La masa factible predice la ocupación real casi perfectamente, y se calcula solo con N.
(No así la posición: el cuello *central* está vacío en varios puntos. La concentración
es irregular, no central.)

### La receta transferible

Ajustada **solo con N = 8, 9, 10, 11**, y aplicada como filtro del conjunto factible
antes de reconstruir R2/M₃/M₄/R5:

```
visitar m ∈ {3,4,5,6}; en cada punto quedarse con los cuellos que
acumulan el 93% de la masa factible
```

| N | factibles | filtrado | R2+R5 solas | con la receta | órbitas |
|---|---|---|---|---|---|
| 10 | 762 | 657 | 0.188% | **0.086%** | 95.7% |
| 11 | 3 027 | 2 662 | 1.660% | **0.792%** | 97.4% |
| 12 | 13 153 | 11 367 | 1.418% | **0.737%** | 94.3% ← fuera del ajuste |
| 13 | 59 219 | 46 990 | 6.623% | **1.995%** | 98.3% ← fuera del ajuste |

**El espacio se reduce a la mitad o a un tercio, y la cobertura de órbitas se mantiene
entre 94% y 100% en los seis N**, incluidos los dos que no participaron en el ajuste.
Quitar los cuellos raros elimina solo el 13–21% de los espectros factibles, pero son
justo los que estiraban la envolvente.

Es lo primero de todo el proyecto que **rompe el patrón «exacto e inútil»**: es
deliberadamente inexacto (pierde el 10% de las soluciones) y a cambio poda de verdad.

### Los caminos

Tratando el mapa como ciudades (puntos) y barrios (cuellos), y haciendo caminos hasta
cubrir una solución de cada órbita:

| N | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|
| órbitas | 12 | 46 | 92 | 341 | **1787** |
| saltos (ponderado por masa) | 3 | 4 | 3 | 3 | **8** |
| saltos (uniforme) | 4 | 9 | 10 | 7 | 17 |

Las órbitas se multiplican por 149 y el camino pasa de 3 a 8. Pero eso es también su
límite: **cubrir es fácil porque las celdas son enormes** (la mediana es el 7% del
espacio), y cubrir barato no poda.

Y no hay carretera secundaria: `corr(tráfico, densidad de soluciones)` es **positiva**
en los cuatro N. Las celdas poco transitadas están vacías, no son ricas.

→ `necks.py`, `paths.py`, `fig_necks.png`, `fig_paths_map_n10.png`, `fig_paths_density.png`

## El invariante en el árbol · `inv_prune.py`

R5 sobre tableros a medio llenar. Lo que **no** funciona, con su razón:

- El invariante de lo **puesto**: AUC 0.56 / 0.58. Apenas por encima del azar.
- Su versión exacta: el techo por retardo `max A(k)` sobre los factibles resulta ser
  **exacto lag a lag** (coincide con el máximo real en los 18 retardos de N=10) y aun
  así poda **0.0%** de los nodos. El máximo lo alcanza alguna solución.

Lo que sí: el invariante de lo **pendiente** — la autocorrelación de las diagonales
aún disponibles.

| rasgos | AUC N=10 | AUC N=11 |
|---|---|---|
| la fórmula de 7 rasgos | 0.906 | 0.899 |
| invariante de lo puesto | 0.559 | 0.575 |
| **invariante de lo PENDIENTE** | 0.856 | 0.852 |
| **fórmula + pendiente** | **0.944** | **0.935** |

+3.8 puntos con un modelo lineal de 29 rasgos. En poda real (entrenado en N=10 y 11,
medido en N=12), a igual conservación de soluciones gana **7–12 puntos** de ahorro de
nodos, y a conservación total pasa de 1.9% a 11.0%.

**Los picos como heurística** (razones entre los que hay y los que faltan) dan AUC
0.71 — señal real, no artefacto (la profundidad sola da 0.510, y a profundidad fija
suben a 0.76–0.79). Pero ningún cociente aguanta solo (todos entre 0.48 y 0.57), y
sobre la autocorrelación pendiente aportan **+0.001**. Redundantes.

→ `inv_prune.py`, `fig_inv_prune.png`, `fig_inv_peaks.png`

## La lectura física · `slits.py`, `sphere.py`

Toda la sesión ha estado calculando **patrones de difracción** sin llamarlos así. Con
rendijas puntuales en las diagonales ocupadas, el campo lejano es

```
I(x) = |Σᵢ exp(i·dᵢ·x)|² = |W(x)|²
```

que es el testigo, sin cambiar una letra. El diccionario:

| óptica | N-reinas |
|---|---|
| rendija | diagonal ocupada |
| intensidad media | `N + 2·conflictos` |
| franja brillante | una diferencia de diagonales repetida |

**Lo que una solución minimiza es la POTENCIA, no las franjas.** Dos reinas en la
misma diagonal no son dos rendijas: son **una con amplitud 2**, que transmite 4 veces
la potencia. La condición de N-reinas es repartir las N unidades de amplitud lo más
finamente posible.

Las franjas no se minimizan: `Σₖ A(k) = C(N,2)` es **fija**. Para no tenerlas haría
falta un conjunto de Sidon (`A(k) ≤ 1`), y es geométricamente imposible: N marcas en
2N−1 huecos es densidad 1/2, y un Sidon en M huecos admite ~√M marcas. Medido sobre
todas las aperturas válidas de N=8: `max A(k)` va de 4 a 7, nunca cerca de 1.

Y da lectura física a la pared: **una pantalla no puede distinguir dos aperturas con
la misma autocorrelación**. Las 13 parejas de órbitas de N=9 son parejas de aperturas
ópticamente indistinguibles. Es recuperación de fase, un problema conocidamente
irresoluble en general.

### En 3D

Con emisores sobre una esfera, la potencia radiada tiene la misma forma cerrada con el
núcleo cambiado: **la delta de 1D se convierte en un sinc**.

```
1D:   ⟨I⟩ = N + 2·Σ δ(dᵢⱼ)          ≥ N siempre
3D:   P   = N + 2·Σ sinc(k·dᵢⱼ)     puede bajar de N
```

Y esa tercera opción —`sinc < 0`, el par *resta*— no existe en 1D. Relajando 12
emisores se llega a **P/N = 0.45** en `kR = 4`: subrradiancia de Dicke. En 3D hay
equilibrio de verdad; en N-reinas solo hay restricción.

| kR | 0.5 | 1 | 2 | **4** | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| P/N | 11.03 | 8.50 | 2.48 | **0.45** | 0.53 | 0.74 | 0.90 |

Con `kR` grande hay `(kR)² ≫ N` canales que apagar y `P/N → 1`: **N-reinas es el
límite de longitud de onda corta del problema esférico.**

Lo que **no** funciona: emisores estáticos (el campo con todos encendidos es una sola
función fija, no depende de ninguna solución) y emisores relajando libremente. Esto
último con número: sobre matrices doblemente estocásticas, el estado **difuminado**
`1/N` radia **menos** que cualquier solución (10.75 contra 16.0 en N=8), porque
minimizar `Σc²` con `Σc` fijo significa repartir, no concentrar en 0/1. Una solución
es un **vértice de energía alta**, no un mínimo. Es el mismo muro de `newton_queens/`
y `wave_queens/`, ahora explicado: por eso hacían falta el termostato y la
heterogeneidad.

→ `slits.py`, `sphere.py`, `fig_slits.png`, `fig_slits_interf.png`, `fig_sphere.png`

## Los niveles · `delsarte.py`

La única familia que quedaba sin probar. En vez de *describir* las soluciones,
**acotar cuántas puede haber**.

### Nivel 1: marginales. Captura 0%

```
log Q(N) ≤ max { Σᵢⱼ h(pᵢⱼ) }   sobre filas=1, columnas=1, diagonales ≤ 1
```

Programa convexo; el dual se resuelve por descenso coordenado exacto. Resultado:
`α ≈ −0.98`, **peor que la cota trivial de N!**, y las diagonales aportan
exactamente 0.0000.

La causa es de una línea: **el óptimo es `pᵢⱼ = 1/N` exacto**, y ahí la diagonal más
larga suma `N·(1/N) = 1`. Justo el límite. Las restricciones de diagonal están
**inactivas** — 1 de 39 activa en N=20, y esa marginalmente. La relajación de primer
orden es **ciega a las diagonales**, que son todo el problema.

### Nivel 2: pares. Captura 70.6%

`Q(N) ≈ N!·e^{−E[C]}` con `E[C] = (2N−1)/3` exacto (choques esperados en una
permutación al azar):

| N | 10 | 16 | 20 | 27 |
|---|---|---|---|---|
| α real | 1.6441 | 1.7408 | 1.7764 | 1.8144 |
| α de pares | 1.4255 | 1.5014 | 1.5290 | 1.5591 |
| α de N! | 0.7921 | 0.8556 | 0.8790 | 0.9048 |
| **captura** | 74.3% | 73.0% | 72.4% | **71.9%** |

Asintóticamente `α_{N!} = 1`, `α_pares = 1 + 2/3 = 1.667`, `α_real = 1.944` (Simkin
2021). El nivel de pares captura **(1.667−1)/(1.944−1) = 70.6%**.

### El 30% que falta: correlación

Los choques no son independientes. Sobre 200 000 permutaciones:

```
N=10   Var[C]/E[C] = 1.0260        (Poisson daría 1.000)
N=27   Var[C]/E[C] = 1.0236
```

Positivamente correlacionados — el mismo hecho que la tabla de independencia del
frente 1. La corrección de segundo cumulante no basta; hace falta el desarrollo
completo de cúmulos, que es el trabajo de Simkin.

**El mapa de la dificultad, cuantificado:**

```
nivel 1 (marginales)        →   0%
nivel 2 (pares)             →  70.6%
nivel 3+ (correlaciones)    →  29.4%
```

→ `delsarte.py`

---

# Conclusión de la segunda sesión

Lo mejor que salió: **el invariante de simetría** (exacto, gratis, y hace el cociente
justo por D4), **la masa factible como predictor** de los cuellos (r ≈ 0.997,
derivable, y con transferencia verificada fuera del ajuste), y **el mapa de niveles**
—0%, 70%, 30%— que localiza la dificultad.

La regla que resume el proyecto entero, ahora con evidencia de tres frentes
independientes:

> **La dificultad de N-reinas no vive en las marginales, vive en los pares.**
> Todo lo que ha funcionado —el testigo, el invariante, R5, la cota de nivel 2— es de
> pares. Todo lo que ha fallado —momentos, perfiles, marginales, la cota de nivel 1—
> es de primer orden. Y el 30% que ni los pares capturan es correlación entre pares.

Y la razón por la que ninguna traslación disolvió el problema, dicha con precisión:
si puedes traducirlo a otro espacio en tiempo polinómico, **la traducción es una
reducción**, y las reducciones preservan la dureza. Completar un tablero de N-reinas
es NP-completo (Gent, Jefferson y Nightingale, JAIR 2017), así que cada regla de poda
de este proyecto es una aproximación a un oráculo NP-completo. Por eso todas se
estancan y ninguna llega al 100%: no les falta ingenio, les falta resolver un
problema NP-completo.

Con la asimetría que eso implica, y que conviene no olvidar:

- **Encontrar una solución es fácil** — `O(N)`, con fórmula cerrada. Las familias
  `σ(i) = (a·i+b) mod N` y la rotación de Sturm de `sturmian_queens/` lo hacen.
- **Listarlas todas no tiene patrón**, y no por falta de ingenio: está prohibido salvo
  que P=NP. Medido: la familia con fórmula cubre el 70% de Q(7), el 3.3% de Q(11), el
  0.18% de Q(13) y el **0%** de Q(16). Y la única aplicación afín que preserva el
  conjunto de soluciones es el espejo, que ya estaba en D4.

**Dónde seguiría**: geometría de polinomios y polinomios reales-estables — el área que
resolvió van der Waerden sobre permanentes, las conjeturas de Mason y Kadison–Singer.
`Q(N)` es un permanente restringido, y los permanentes son su objeto central. Lo que
evitaría es más geometría descriptiva: hélices, toros, curvas polares, difracción.
Describen con enorme precisión y no construyen nada. Lo que funciona es lo que
**cuenta** o **acota**, no lo que **dibuja**.

---

## Ejecutar (sesión 2)

```bash
venv/bin/python -m diag_waves.pp          # el embudo R2/M3/M4/R5
venv/bin/python -m diag_waves.orbit_geo   # el invariante de simetria y las colisiones
venv/bin/python -m diag_waves.necks       # los cuellos y la receta transferible
venv/bin/python -m diag_waves.paths       # los caminos por el mapa de cuellos
venv/bin/python -m diag_waves.inv_prune   # el invariante pendiente en el arbol
venv/bin/python -m diag_waves.slits       # N-reinas como difraccion
venv/bin/python -m diag_waves.sphere      # emisores en una esfera, subrradiancia
venv/bin/python -m diag_waves.delsarte    # la cota de entropia y los niveles
```
