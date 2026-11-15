# camino_queens — un orden de los N^N tableros que junta las soluciones

Idea: listar los N^N tableros (x_i = fila de la reina de la columna i) en un
orden escogido de forma que las soluciones queden en una zona cerrada del
listado, y saber dónde empieza esa zona.

Un **camino** es un árbol: el nivel k lo decide un funcional lineal
f_k(x) = Σ w_i x_i (mod N), y el índice del tablero es el número en base N con
dígitos (f_1, …, f_N), f_1 el más significativo. Con N funcionales
independientes es una biyección de los N^N tableros. Un grupo de soluciones
cae en un único subárbol de profundidad k si f_1..f_k son constantes sobre él.
La profundidad máxima posible es N − (dimensión afín del grupo).

```
cc -O2 -o camino camino.c
./camino 5        # N primo, 5..13; salidas en data/salida_N*.txt
```

Órdenes que compara:

- `LEX`: el orden normal, f_k = x_{k−1}.
- `D2`: segundas diferencias arriba, f_k = x_{k−1} − 2x_k + x_{k+1}
  (k = 1..N−2), después a = x_1 − x_0 y por último b = x_0.
- `MOM`: arriba los momentos Σ x_i, Σ i·x_i, Σ i²·x_i; debajo, segundas
  diferencias hasta completar N−2 niveles; después a y b.

## Resultados (octubre 2026)

### 1. Toroidales lineales (x_i = a·i + b): un bloque perfecto en todo N primo

| N | lineales | LEX: extensión | D2: posiciones | densidad |
|---|---|---|---|---|
| 5 | 10 | 2.409 de 3.125, 10 tramos | 10 … 19 | 1 |
| 7 | 28 | 732.841, 28 tramos | 14 … 41 | 1 |
| 11 | 88 | 2,7·10¹¹ | 22 … 109 | 1 |
| 13 | 130 | 2,9·10¹⁴ | 26 … 155 | 1 |

- **El subárbol:** las segundas diferencias son 0 exactamente en las rectas
  x = a·i + b. Las N−2 cifras de arriba valen 0, así que el bloque está en el
  primer subárbol, que contiene N² tableros.
- **El inicio:** dentro de ese subárbol el índice es a·N + b. El bloque empieza
  en **2N**, con la solución x_i = 2i, y termina en (N−1)N − 1.
  - Antes del bloque están las rectas de pendiente 0 (todas las reinas en la
    misma fila) y de pendiente 1 (todas en la misma diagonal).
  - Después está la pendiente N−1 (la otra diagonal).
  - Es decir, el bloque son las rectas que no fallan.
- **Sin una combinación de columnas no funciona:** en LEX ningún nivel es
  constante, y las soluciones quedan dispersas por todo el listado.

**Hay muchos caminos, pero todos usan el mismo subárbol.** El número de caminos
salió por enumeración exhaustiva; que todos usen el mismo subárbol está
probado por álgebra, no por búsqueda:

- Para que el grupo quepa en un subárbol de profundidad N−2, los niveles de
  arriba tienen que anular las direcciones (1,…,1) y (0,1,…,N−1).
  - En N=5, de los 3.124 funcionales posibles para el primer nivel solo 124
    cumplen esto (= 5³ − 1).
  - En N=7, 16.806 de 823.542.
- Dentro del subárbol se enumeraron todas las elecciones de los dos últimos
  niveles (con traslaciones de valores).
  - Las que dejan las soluciones en un bloque contiguo son **8·(N−1)·N²**:
    800, 2.352, 9.680 y 16.224 para N = 5, 7, 11, 13.
  - Ninguna da una progresión aritmética con salto mayor que 1: las que dan
    progresión son exactamente las contiguas.
  - 8 = 2 orientaciones de a × 4 posiciones del intervalo; N²(N−1) = las
    transformaciones afines de b.
  - El número coincide con el tamaño del grupo de simetrías de las toroidales,
    8(p−1)p²; no se ha comprobado si es casualidad.

### 2. Toroidales no lineales (N=13): un subárbol de profundidad 3, nada más

N=13 tiene 4.524 toroidales, de las que 4.394 no son lineales. Su dimensión
afín es 10, y los 3 funcionales constantes son exactamente los momentos:

```
Σ x_i ≡ Σ i·x_i ≡ Σ i²·x_i ≡ 0   (mod p)   para toda solución toroidal
```

Salen de que x, x+i y x−i son permutaciones de Z_p: con las sumas de potencias
1, 2 y 3, la parte lineal se cancela con esos tres pesos. Con la potencia 4
aparecen términos no lineales, así que no hay más momentos lineales.

- En el orden `MOM`, las 4.524 caen en el primer subárbol de profundidad 3,
  que es 13³ = 2.197 veces más denso que la media.
- Dentro de ese subárbol quedan repartidas en 313 tramos. Son las 348 órbitas
  de desplazamiento b → b+1 (13 soluciones consecutivas cada una), y algunas
  se juntan entre sí. Las 130 lineales siguen formando un bloque, anidado
  dentro.
- No hay ningún nivel lineal más que las comparta: la dimensión afín es 10,
  así que con funcionales lineales no se baja de ahí.

### 3. Todas las toroidales (todos los niveles): un camino por centro

Programas: `grado.c` (base de Newton), `soporte.c` (base de monomios),
`simetria.c`. Salidas en `data/grado_p*.txt`, `data/soporte*_p*.txt` y
`data/simetria_p*.txt`.

```
cc -O2 -o soporte soporte.c && ./soporte 17        # ./soporte 17 rot: también con los 8 giros
```

**Base de Newton.** Se ordena por las diferencias c_k = Δ^k x(0), con la
mayor arriba. Es el orden D2 generalizado: una solución de grado D cae en el
primer subárbol, de p^(D+1) tableros. El grado solo toma unos pocos valores, y
es el mismo en las 8 imágenes por los giros:

| p | grados que aparecen (soluciones) |
|---|---|
| 13 | 1 (130), 7 (1.352), 9 (3.042) |
| 17 | 1 (238), 9 (4.046), 11 (2.312), 12 (36.992), 13 (97.104) |
| 19 | 1 (304), 10 (5.776), 13 (8.664), 14 (51.984), 15 (753.768) |

Pasar del grado 1 al siguiente multiplica el tamaño del subárbol por 10⁶ o
más, así que esta base no sirve para juntar las que no son lineales.

**Base de monomios.** Se escribe x(i) = Σ a_k i^k. Si la solución tiene
**centro** en la columna 0, es decir, una simetría x(ζi) = ζ^r·x(i) con ζ de
orden m, entonces solo usa exponentes k ≡ r (mod m).

- Ese conjunto de exponentes U es un camino: arriba van los coeficientes fuera
  de U (que valen 0) y todas esas soluciones caen en el primer subárbol, de
  p^(|U|+1) tableros.
- Cada columna centro da otro camino: se desplazan las columnas para que el
  centro quede en 0.

Caminos en p=13, con el centro en la columna 0 (soluciones en el subárbol /
tamaño del subárbol):

| U | qué recoge | soluciones / tableros |
|---|---|---|
| {1} | lineales | 130 / 169 |
| {7} | x = a·i·χ(i): dos espirales cruzadas en 0 (χ = residuo cuadrático) | 52 / 169 |
| {1,7} | dos espirales, subgrupo de los cuadrados | 234 / 2.197 |
| {1,5,9} | exponentes ≡ 1 (mod 4), subgrupo μ4 | 208 / 28.561 |
| {1,3,5,7,9} | impares: cualquier solución con centro en 0 | 468 / 4,8·10⁶ |

**En p=13 los 13 caminos impares (uno por columna centro) recogen las 4.524
toroidales.** Las subfamilias van anidadas dentro, y la densidad respecto al
azar se multiplica por unos 6·10⁶.

**Más allá de p=13 no alcanza.** Proporción de toroidales según su simetría
(`simetria.c`):

| p | con centro (las recoge un camino) | otra simetría (giro de 90° o 45°) | ninguna simetría |
|---|---|---|---|
| 13 | 100 % | 0 | 0 |
| 17 | 34,3 % | 39,4 % | 26,3 % |
| 19 | 24,0 % | 0 | 76,0 % |

- **Giros de 90° o 45° (39 % en p=17).** Existen porque p ≡ 1 (mod 4). Sus
  direcciones propias tienen pendiente ±4, que no es fila, columna ni
  diagonal. En esas coordenadas la solución no es una función, así que no
  hay soporte disperso en ningún marco. Una idea pendiente: usar como
  niveles los momentos Σ u^a v^b en esas coordenadas, que la simetría
  anularía.
- **Sin simetría.** Ningún camino de este tipo puede juntarlas, porque la
  dispersión del soporte sale justo de la simetría. Son la mayoría desde
  p=19, y se espera que la proporción crezca con p.
- **Una trampa a evitar.** Si se permite que los niveles cuenten reinas por
  fila y diagonal, las toroidales quedan trivialmente en un bloque perfecto,
  porque esos niveles son las propias reglas. Por eso aquí solo se usan
  lecturas del tablero como polinomio.

### 4. Muchos caminos generados, contra un conjunto falso (`grupos.c`)

Hipótesis: cada camino agrupa su propia sección de soluciones, y generando
muchos alguno puede identificar un grupo por casualidad.

- **Cómo se genera un camino.** Un camino lineal es una cadena de
  subespacios F_0 ⊂ F_1 ⊂ …, uno por nivel del árbol. Se elige una solución
  base al azar y, en cada nivel, se añade la dirección que más soluciones
  recoge. n_k = soluciones en el subárbol del nivel k, que tiene p^k
  tableros.
- **El conjunto falso.** Es el mismo proceso, con el mismo número de caminos y
  el mismo tamaño, sobre permutaciones al azar que cumplen las mismas sumas
  (Σi·x ≡ Σi²·x ≡ 0) y las mismas simetrías afines. Todo se trabaja con
  x_0 = 0 (es decir, sin contar el subir o bajar todas las reinas a la vez).

```
cc -O2 -o grupos grupos.c && ./grupos 17 n 2000     # clase: c centro, o otro giro, n sin simetría, t todas
```

Máximo de n_k: real / falso (salidas en `data/grupos_p*.txt`; repetido con el
desempate corregido, ver la sección 6):

| caso | k=3 | k=6 | k=9 | k=11 | penúltimo nivel | lectura |
|---|---|---|---|---|---|---|
| p=13, todas | 26 / 8 | 52 / 22 | — | — | 132 / 73 (k=8) | recupera los grupos con centro (36 en k=5) |
| p=17, con centro | 42 / 6 | 84 / 14 | 252 / 25 | 404 / 64 | 552 / 272 | grupos claros, de 6 a 11 veces el azar |
| p=17, giro 90°/45° | 8 / 6 | 24 / 14 | 44 / 26 | 108 / 72 | 400 / 306 | casi como el azar |
| p=17, sin simetría | 6 / 6 | 14 / 14 | 32 / 24 | 220 / 62 | **2176 / 230** | un único grupo global, ver abajo |
| p=19, sin simetría | 6 / 6 | 14 / 12 | 22 / 24 | 36 / 36 | 2034 / 2010 | **igual que el azar en todos los niveles** |

- **Con centro:** la búsqueda a ciegas encuentra sola los caminos de la
  sección 3. El método funciona.
- **p=17 sin simetría:** las 36.992 caben todas en un subárbol un nivel más
  profundo que el resto. Cumplen una cuarta suma, Σ i³·x_i ≡ 0, y coinciden
  en número exacto con las de grado de Newton 12 (36.992). Es un grupo real,
  pero de un solo nivel.
- **p=19 sin simetría (76 % de las toroidales):** ningún camino lineal de los
  600 generados se separa del conjunto falso. Con lecturas lineales del
  tablero, estas soluciones son indistinguibles de permutaciones al azar con
  las mismas sumas.
- **Límite del método:** la generación es codiciosa y puede saltarse grupos
  que solo aparecen eligiendo mal al principio. Aun así, encontró todos los
  grupos conocidos.
- **Siguiente paso:** probar otras lecturas no lineales del tablero (xᵢ²,
  xᵢ·xⱼ, momentos en las coordenadas de los giros), cada una con su propio
  conjunto falso.

### 5. Caminos sobre lecturas no lineales (`grupos2.c`)

Una lectura es una función φ: tablero → Z_p^D. El camino es la misma cadena
codiciosa de subespacios, pero en el espacio de φ; en el tablero, cada nivel
es una ecuación polinómica. Hay tres lecturas:

- `sep L`: Σ c^j·x_c^l con l ≤ L. Con `sep 1` se recupera exactamente la
  lineal de `grupos.c`.
- `tot d`: Σ c^j·x_c^l con j + l ≤ d. Genera el mismo espacio que los
  momentos Σ u^a v^b en las coordenadas propias de cualquier giro, y un
  camino no depende de la base, así que cubre las coordenadas de los giros.
- `quad`: x_a y todos los productos x_a·x_b.

```
cc -O2 -o grupos2 grupos2.c && ./grupos2 17 o tot 6 400 60
```

**Dos conjuntos falsos.** En estas lecturas las toroidales cumplen muchas más
ecuaciones globales que las permutaciones falsas (por ejemplo 15 frente a 8 en
`tot 6`). Esas ecuaciones salen de las reglas, igual que Σx, y hacen que las
reales parezcan agrupadas sin estarlo. Por eso se añade un segundo falso:
puntos al azar dentro de la envoltura de φ(reales), el mismo subespacio
afín. El cociente que cuenta es **real / azar en la envoltura**.

Máximo de ese cociente sobre los niveles (salidas en `data/g2_*.txt`, con el
desempate ya corregido):

| caso | tot 6 | tot 10 | sep 2 | quad |
|---|---|---|---|---|
| p=17 con centro | 252 soluciones con φ idéntico (nivel 0) | 38× | 17× | 3,0× |
| p=17 giro 90°/45° | 29,5× (384 en nivel 3, frente a 13) | 31× (pero media 30 frente a 20) | 1,7× | 1,0 |
| p=17 sin simetría | 3,8× | 1,0 | 1,0 | 1,0 |
| p=19 sin simetría | 4,4× (114 frente a 26) | 1,0 | 1,1 | 1,0 |

Este falso todavía no está cerrado por las simetrías. La sección 6 repite las
comparaciones con uno que sí lo está, y la mayor parte de estos excesos
desaparece.

`quad` no da nada, con una reserva: la envoltura tiene dimensión 87–132 y
el camino solo se exploró hasta 40–60 niveles.

### 6. Falso cerrado por las simetrías (`grupos3.c`), y un error corregido

**El error.** Cuando varias direcciones empataban, el desempate favorecía las
últimas del array. En el conjunto real, las entradas consecutivas eran la
misma solución con todas las reinas subidas 0, 1, 2… filas, así que el camino
completaba esa órbita siempre. De ahí salía el "12× en p=19 con `sep 2`" que
se publicó en una versión anterior de esta sección: con el desempate uniforme
y los arrays barajados, baja a 1,1. Los tres programas están corregidos y
`rerun.sh` repite todo.

**El falso con simetrías.** Se usa el conjunto completo (todas las filas b,
sin normalizar x_0). El grupo H (desplazar columnas, subir filas, escala
(i,x)→(ui,ux), x→−x) actúa linealmente sobre los momentos M_{j,l}; las
fórmulas están en la cabecera de `grupos3.c` y se comprueban contra la
transformación directa del tablero. Hay dos versiones:

- **Envoltura con simetrías:** tantos puntos al azar en la envoltura como
  órbitas tienen las reales, y de cada uno su órbita completa por H.
- **Tableros con simetrías:** permutaciones al azar cuyo φ cae en la envoltura
  real, con su órbita por H. Así se respeta también que φ venga de un
  tablero. En `tot 6` no se encuentran (demasiadas ecuaciones); en `sep 2`
  sí.

```
cc -O2 -o grupos3 grupos3.c && ./grupos3 17 n tot 6 150 12
```

Máximo de n_k, real frente al falso con simetrías (`data/g3_*.txt`):

| caso | frente a la envoltura uniforme | frente a la envoltura con simetrías |
|---|---|---|
| p=17 con centro, `tot 6` | 2.142× en nivel 0 | **2.142× en nivel 0, 97× en nivel 2: sobrevive** |
| p=17 giro 90°/45°, `tot 6` | 28× | 0,5–1,25×: **desaparece** |
| p=17 sin simetría, `tot 6` | 20× | 2× constante desde el nivel 0 |
| p=19 sin simetría, `tot 6` | 14× | máximo 4,75×, pero media 1,5× |
| p=17 sin simetría, `sep 2` | 1,0 | igual o por debajo (0,2–0,9×) |
| p=19 sin simetría, `sep 2` | 1,1 | en media 7,4 frente a 7,0 (nivel 6) y 15,5 frente a 13 (nivel 12) |

- **El 2× de las asimétricas de p=17** aparece ya en el nivel 0: cada solución
  real tiene otra con φ idéntico. Probablemente sea un giro del grupo completo
  (que no está en H) que no cambia los momentos de grado ≤ 6; no se ha
  comprobado.
- **El residuo de p=19 en `tot 6`** (media 1,5×, máximo 4,75×) es pequeño y no
  se ha podido comparar con tableros falsos.

**Conclusión de las secciones 4 a 6.** Generando muchos caminos, lineales y no
lineales, solo hay un tipo de grupo que resiste a todos los falsos: **los de
las soluciones con centro**. Las del giro y las asimétricas quedan explicadas
por la estructura de las órbitas de simetría, por las ecuaciones globales de
las toroidales o, en un caso, por mi error de desempate. Queda un residuo
pequeño en p=19 (`tot 6`) que no ha llegado a ser un grupo identificable.

### 7. Trayectorias: saltar de solución en solución (`trayectorias.c`)

Se relajan las reglas del camino: no hace falta recorrer todos los tableros,
se puede repetir, y la trayectoria puede pasar por soluciones que no cuentan.
Una trayectoria es una regla corta b_0, b_1, … y se puntúan **solo los pasos
donde predice una solución**:

| patrón de salto | trayectoria | la fijan |
|---|---|---|
| salto constante | recta b_k = s + k·v | 2 soluciones |
| salto que crece aritméticamente | parábola b_k = s + k·v + C(k,2)·w | 3 soluciones |

Solo cuentan los aciertos además de los que fijan la regla. Un acierto por
azar es casi imposible (probabilidad ~ n/p^(p−3)), así que todo lo que aparece
es estructura; la pregunta es si se explica por simetrías.

Cómo se calcula:

- Todo en el cociente por "subir filas" (x_0 = 0), para que esa simetría no
  regale rectas. Se descartan las parábolas degeneradas.
- H lleva rectas en rectas y parábolas en parábolas, así que basta una base
  por órbita.
- Las bases son de la clase estudiada; los aciertos cuentan en **todas** las
  toroidales.
- Falso: permutaciones con las mismas sumas, en órbitas por H.

```
cc -O2 -o trayectorias trayectorias.c && ./trayectorias 17 n 0     # pares 0 = todos
```

| caso | rectas con ≥3 | parábolas con ≥4 (por par) | parábolas con ≥5 | falso |
|---|---|---|---|---|
| p=13, todas | 3 sol: 7,2 por base; 10 sol (las lineales) | 13,9 (hasta 7 soluciones) | sí | casi nada (0,035) |
| p=17, con centro | 3 sol: 18 por base; 14 sol | 7,6 (hasta 7) | sí | ninguna |
| p=17, giro | ninguna | 0,045, todas del tipo `ccoo` | ninguna | ninguna |
| p=17, sin simetría (todos los pares) | ninguna | 0,075, todas del tipo `ccnn` | ninguna | ninguna |
| p=19, sin simetría | ninguna | 0,17, todas del tipo `ccnn` | ninguna | ninguna |

(`ccnn` = la parábola toca dos soluciones con centro y dos sin simetría.)

**Esas parábolas de 4 se explican por simetrías.** Se comprobó en las 340 de
la muestra de p=17 y, para la primera parte, en las 804 de p=19:

- Las dos asimétricas son siempre s y −s (cambiar el signo de las filas, que
  está en H).
- Las dos con centro son siempre c y λ·c, con λ ∈ {2, 4, 8, 9, 13, 15}: las
  potencias de 2 mod 17, es decir, la propia rotación de orden 8 de esas
  soluciones. La parte λ·c no se comprobó en p=19.
- Los cuatro puntos están en dos rectas que pasan por el tablero "todas en la
  fila 0". Dos rectas así siempre comparten un plano, y en ese plano casi
  siempre hay una parábola que pasa por los cuatro.
- Es geometría que sale de dos simetrías, no un patrón de salto nuevo.

**Conclusión.** Con saltos constantes o que crecen aritméticamente,
permitiendo soluciones intermedias que no cuentan:

- las soluciones con centro forman trayectorias largas, de hasta 14
  soluciones en recta y 7 en parábola;
- las del giro y las asimétricas no tienen ni una recta de 3 ni una parábola
  de 5, y las de 4 que aparecen vienen de simetrías.

El patrón geométrico de verdad, b_{k+1} = M·b_k + c, está en la sección 8.

### 8. Saltos geométricos: b_{k+1} = M·b_k + c (`saltos.c`)

"Cada d pasos" con M es lo mismo que un paso con M^d, así que basta buscar
cadenas s_0 → s_1 → s_2 → … con una misma G(b) = M·b + c. Con M general
cualquier cadena corta se ajusta, así que se usan dos familias pequeñas:

- **Diagonal:** G(x)_i = λ_i·x_i + c_i. El salto de cada columna se
  multiplica por λ_i en cada paso: es el salto geométrico columna a columna.
  La fijan 3 soluciones; cuenta desde la 4.ª.
- **Permutación de columnas:** G(x) = x∘π. La fijan 2 soluciones; cuenta
  desde la 3.ª. Se excluyen las traslaciones de columnas, que son simetrías.

En ambas, las soluciones intermedias de la trayectoria no cuentan.

```
cc -O2 -o saltos saltos.c && ./saltos 17 n 0
```

Para cada cadena se mira si **algún par** de sus soluciones está relacionado
por el grupo completo de simetrías (H y los 8 giros); salidas en
`data/saltos_*.txt`:

| clase | diagonal | permutación de columnas | explicadas por simetría | falso |
|---|---|---|---|---|
| p=17 sin simetría (todos los pares) | ninguna | 136 cadenas de 3 (`cnn`) | **136 de 136** | nada |
| p=19 sin simetría | ninguna (muestreado) | 62 de 3 (`cnn`, `ccn`; la permutación se recorrió entera) | **62 de 62** | nada |
| p=17 giro | ninguna | 349 de 3, 56 de 4 | 364 de 405 | nada |
| p=17 con centro | hasta 13 (escalar filas: c, λc, λ²c…) | hasta 8 | 810/810 de escalar filas; 274/370 con λ distintos; 5468/7118 con permutación | nada |

- **Asimétricas:** no hay ningún salto geométrico por columna que encadene 4
  soluciones. Todas las cadenas de 3 por permutación de columnas se explican
  por simetría. Ejemplo: s → 4·i (una lineal) → s′, donde s′ es la
  transpuesta de s corregida por escalas.
- **Giro:** quedan 41 cadenas sin explicar. En los ejemplos, pasar de una a
  otra es intercambiar las reinas de pocas columnas (3↔11 y 6↔8), lo que
  encaja con las mezclas e intercambios de bloques conocidos en p=17
  (levitation_queens). No se ha comprobado cadena por cadena.
- **Con centro:** siguen siendo las únicas con trayectorias largas.

**Conclusión de las secciones 7 y 8.** Con trayectorias que saltan de
solución en solución, por rectas, parábolas, saltos geométricos por columna o
permutaciones de columnas, y aunque haya soluciones intermedias que no
cuentan, las asimétricas no forman ninguna cadena que no venga de una
simetría.

### 9. No toroidales: trayectorias en el tablero normal (`notor.c`)

En el tablero normal las cuentas son enteras, sin "mod N", y la única
simetría es D4 (8 elementos). Se prueban las mismas familias de
trayectorias, con soluciones intermedias que no cuentan:

- recta y parábola enteras;
- diagonal (el salto de cada columna se multiplica por λ_i racional);
- permutaciones. Renombrar filas y permutar columnas dan la misma cadena,
  porque s∘(s⁻¹t)^k = (t s⁻¹)^k∘s, así que se tratan como una sola familia.
  Se separan los giros cíclicos (r → r+c mod N, i → i+c mod N), las afines
  mod N y el resto.

Las bases son soluciones no toroidales, una por órbita de las simetrías
afines; los aciertos cuentan en todas las soluciones. Falso: permutaciones al
azar en órbitas por D4, del mismo tamaño. Para cada cadena se mira si algún
par está relacionado por D4, y si es una toroidal de N+1 sin la reina de una
esquina (la construcción clásica).

```
cc -O2 -o notor notor.c && ./notor 12 0 15       # N, bases (0 = todas), pares muestreados
GIROS=1 ./notor 14 0 0                            # solo los giros de filas
```

**Ojo con las permutaciones:** aquí el azar no es cero. Hay N! permutaciones
frente a N^N tableros, y el falso ya da cadenas de 3 (100 en N=11, 1.556 en
N=12).

| N | recta | parábola ≥4 | diagonal | girar filas (r+c mod N) | otra permutación: real / falso |
|---|---|---|---|---|---|
| 8 | 0 | 1 | 0 | 4 de 3 | 6 / 0 |
| 9 | 0 | 4 | 0 | hasta 5 | 78 / 28 |
| 10 | 0 | 0 | 0 | hasta 4 | 50 / 24 |
| 11 | 0 | 2 | 0 | hasta 5 | 328 / 100 |
| 12 | 0 | algunas | 0 | hasta 6 (330 cadenas, falso 0) | 3.656 / 1.572; real hasta 6, falso hasta 4 |
| 13 (muestreado) | 0 | 2 | 0 | hasta 5 | 3.368 / 1.342; real hasta 7 |

En la tabla, "4 de 3" quiere decir 4 cadenas de 3 soluciones.

- **Rectas y diagonal:** nada en ningún N.
- **Otra permutación:** el real dobla más o menos al falso, y solo el real
  llega a cadenas de 5 a 7. En N=12, 14 de las 20 cadenas de 5 o más son
  enteras de toroidales de 13 menos una esquina (la construcción conocida).
  El resto no se ha explicado.
- **Afines mod N con a ≠ 1:** ninguna.

**Lo que sí destaca: girar las filas dando la vuelta.** La trayectoria
s, s+c, s+2c, … (mod N), es decir la recta (1,…,1) del toro aplicada a
soluciones normales, encadena muchas soluciones normales. Casi nunca la
explica D4 ni la construcción de la esquina, y el falso no pasa de 2
(`data/notor_giros.txt`):

| N | soluciones en la trayectoria más larga (nº de soluciones con esa racha) | falso |
|---|---|---|
| 9 | 4 (16), 5 (4) | 2 |
| 11 | 4 (12), 5 (4) | 1 |
| 12 | 5 (72), 6 (8) | 2 |
| 13 | 5 (100), 6 (8) | 2 |
| 14 | 7 (520), 8 (248), 9 (152), 10 (24), **11 (16)** | 2 |

Ejemplo de N=14: `0 3 10 7 11 6 2 12 5 8 13 9 4 1` sigue siendo solución al
sumarle 1, 2, …, 10 a todas las filas (mod 14). Es decir, 11 soluciones
normales con salto constante.

- Es el primer camino de este proyecto que agrupa soluciones **sin**
  simetría del problema detrás: el giro de filas no es simetría del tablero
  normal.
- Sí es la simetría del toro, así que lo que mide es lo cerca que está una
  solución normal de ser toroidal.
- N=14 destaca mucho sobre N=12 y 13. Falta entender por qué (¿estructura
  mod 7, o de N+1 = 15?).

### 10. Por qué el giro de filas encadena soluciones, y por qué N=14 (`giros.c`)

**El mecanismo.** Al girar las filas c posiciones, un par de reinas solo
cambia si exactamente una da la vuelta. Entonces choca si |x_i − x_j| = N − d,
con d = |i − j|: es un par que se ataca en el toro pero no en el tablero, un
**par de costura**. Ese par prohíbe una ventana de N − d giros seguidos
(c ∈ [N − b, N − a), con a < b sus filas). Sea W(s) el número de pares de
costura; W = 0 equivale a que s sea toroidal. La trayectoria s, s+1, s+2, …
vale hasta la primera ventana prohibida. `giros.c` comprueba esta regla
contra la medida directa en todas las soluciones de N = 8 a 17: **0
discrepancias**.

Las cadenas largas son, por tanto, soluciones **casi toroidales**: pocos pares
de costura, entre columnas muy separadas (ventanas cortas). En N=14, las 192
soluciones con L ≥ 9 tienen todas W = 2, y en 112 de ellas los dos pares
comparten una reina de la columna del borde.

**Cuándo cabe W = 2.** Una toroidal necesita que x_i + i y x_i − i recorran
todos los restos mod N. Contando sumas y sumas de cuadrados de esas dos
familias (Σ(x+i) + Σ(x−i) y Σ(x+i)² + Σ(x−i)² = 4·Σ j²):

- **N par.** Cada familia tiene al menos una coincidencia, así que W = 2
  significa una en cada una, con r − m ≡ N/2 (r el resto repetido, m el que
  falta). Los cuadrados exigen 2·Σj² ≡ 0 o N/2 (mod N). Si 3 | N,
  2·Σj² ≡ N/3, así que **es imposible** (N = 12, 18…). Si 3 ∤ N se cumple.
- **N primo ≥ 5.** Una sola coincidencia en una familia es imposible
  (r ≡ m). Dos en la misma familia y ninguna en la otra exigen que {r₁, r₂}
  y {m₁, m₂} tengan la misma suma y la misma suma de cuadrados mod p, es
  decir, que sean el mismo par: contradicción. **W ≥ 3.**
- **N impar múltiplo de 3.** La condición de los cuadrados ya no anula nada,
  y W = 2 aparece (N = 9, 15).

Resultado: **W = 2 es posible exactamente cuando N ≡ 2, 3, 4 (mod 6)**. Está
demostrado para N par y N primo, y comprobado en N = 8…17. No se ha probado
para impares compuestos coprimos con 6 (25, 35…).

| N | N mod 6 | W mínimo | trayectoria más larga | soluciones con L ≥ 9 |
|---|---|---|---|---|
| 8 | 2 | 2 | 3 | — |
| 9 | 3 | 2 | 5 | — |
| 10 | 4 | 2 | 4 | — |
| 11 | 5 | 3 | 5 | — |
| 12 | 0 | 3 | 6 | 0 |
| 13 | 1 | 3 | 6 | 0 |
| 14 | 2 | 2 | **11** | 192 (5,3·10⁻⁴) |
| 15 | 3 | 2 | **11** | 364 (1,6·10⁻⁴) |
| 16 | 4 | 2 | **11** | 3.536 (2,4·10⁻⁴) |
| 17 | 5 | 3 | 10 | 1.088 (1,1·10⁻⁵) |

- **Respuesta a "¿por qué N = 14?":** es el primer N, después de 12 y 13
  (donde W ≥ 3), en que caben soluciones con solo dos pares de costura.
  N = 15 y 16 se comportan igual.
- **N = 17** vuelve a W ≥ 3: tiene trayectorias largas por ser más grande,
  pero en una proporción entre 15 y 50 veces menor.
- La media de L cae con W en todos los N; por ejemplo, en N=14 pasa de 3,4
  con W = 2 a 1,0 con W = 8.

Salidas en `data/giros_N*.txt`.

### 11. Caminos de decisión: una reina, un salto de rey (`salto_rey.c`, N=5)

Tablero = N reinas en casillas distintas (no hace falta una por columna). Un
salto = mover una reina a una casilla vecina libre (8 direcciones, hasta 8·N
opciones). Con una búsqueda en anchura exacta sobre los C(25,5) = 53.130
tableros de N=5 (todos conectados), salida en `data/salto_rey_N5.txt`:

- **Distancias:** entre soluciones solo hay distancia 4 o 6.
  - Dos soluciones de la misma pendiente (x = 2i + b o x = 3i + b) están
    siempre a 6.
  - Dos de pendiente distinta están a 4, salvo cuatro de los cinco pares
    espejo (x ↔ 4 − x), que están a 6. El quinto par espejo (3 y 6, las dos
    con una reina en la casilla central) está a 4.
  - Como grafo, los saltos de 4 forman K_{5,5} menos esas 4 aristas: 21 pares
    a distancia 4 y 24 a distancia 6.
- **El recorrido más corto por las 10** tiene 36 saltos: 9 tramos de 4, el
  mínimo posible. Alterna pendientes: 9 8 7 6 4 5 3 2 1 0.
- **Cada tramo** mueve 4 reinas distintas un paso cada una, y la quinta se
  queda quieta. Las mueve en parejas opuestas (↑↓ y ←→, o las cuatro
  diagonales): es intercambiar dos filas vecinas y dos columnas vecinas.
  Ningún tablero intermedio es solución.

### 12. Explorar todos los caminos con memoria (`explora.c`)

Se empieza en una solución y se salta a cualquiera de los 8·N vecinos, en
orden aleatorio. Cada tablero se guarda una sola vez en una tabla hash, con su
padre y su profundidad: nunca se procesa dos veces, y el camino a cualquier
tablero se reconstruye siguiendo los padres. Se para cuando han aparecido
todas las soluciones.

- **Profundidad:** se expande siempre el último tablero descubierto.
- **Anchura:** se expande el más antiguo.

Se repite con inicio y orden al azar: 5.000 veces en N=5 y 200 en N=6
(`data/explora_*.txt`).

| | N=5 (53.130 tableros, 10 soluciones) | N=6 (1.947.792 tableros, 4 soluciones) |
|---|---|---|
| tableros explorados hasta tener todas, en profundidad | 87 % de media (25 %–100 %) | 71 % (12 %–100 %) |
| ídem, en anchura | 65 % (52 %–78 %) | 51 % (47 %–65 %) |
| profundidad en el árbol de cada solución, en profundidad | mediana 1.345 saltos (la distancia real es 4–6) | mediana 36.047 (la real es 6–8) |
| orden de aparición, en profundidad | igual para las cercanas y las lejanas (puesto medio 5,0 y 5,0) | 1,99 frente a 2,02 |

- Con memoria, "todos los caminos" se convierte en explorar el grafo una
  vez, y es finito.
- Pero para reunir todas las soluciones saltando al azar hay que recorrer
  **entre la mitad y casi todo** el espacio de tableros. Las soluciones no
  están en ninguna zona especial del grafo de saltos.
- En profundidad, el camino que lleva a cada solución da cientos o miles de
  vueltas, y el orden en que aparecen no depende de lo cerca que estén. En
  anchura aparecen por distancia, como debe ser.
- N=7 (86 millones de tableros) no cabe así en memoria (unos 4 GB de tabla);
  haría falta guardar los tableros comprimidos.

**Más estadísticas** (5.000 experimentos en N=5, 300 en N=6):

- **Reparto de las soluciones.** Desde una solución, los tableros a
  distancia d forman una campana: en N=5 van de 1, 29, 340… a un máximo de
  15.228 en d=6, y el más lejano está a 10. Las soluciones caen en la zona más
  poblada: 4 en d=4 y 5 en d=6. En N=6 caen en d=6 (2) y d=8 (1), con la
  campana en d=7–8 y el más lejano a 15.
- **Separación:**
  - N=5: solo distancias 4 (21 pares) y 6 (24); bipartito entre las dos
    pendientes.
  - N=6: dos parejas a 8 (0–1 y 2–3); el resto a 6.
- **Profundidad del árbol (el camino más largo).**
  - N=5: mediana 4.846 y máximo 6.202 saltos (el 12 % de todos los tableros
    en un solo camino).
  - N=6: mediana 88.035 y máximo 183.844 (el 9 %).
  - En anchura, el árbol se para en profundidad 6 (N=5) u 8 (N=6).
- **Órdenes de aparición.**
  - En profundidad son aleatorios: 4.998 órdenes distintos en 5.000
    experimentos (N=5), y los 24 posibles con frecuencia pareja en N=6. Las
    secuencias de distancias (4 o 6) salen todas, con frecuencias del mismo
    orden.
  - En anchura solo importa la capa: 4,4,4,4,6,… y, dentro de la capa, el
    orden es aleatorio.
- **Ritmo de aparición, en profundidad.** Las soluciones aparecen a ritmo
  casi constante: en N=5, la mediana entre una y la siguiente va de unos
  3.000 a unos 5.000 tableros.

### 13. Caminos construidos con bits, sin buscar (`salto_bits.c`)

Para ir de la solución A a la B no se explora nada; el camino se construye:

- `A AND NOT B` son las reinas que sobran y `B AND NOT A` las casillas que
  faltan.
- Se empareja cada reina que sobra con una casilla que falta (el
  emparejamiento de menor distancia de rey total).
- Las reinas se mueven de una en una, un salto de rey cada vez, pasando por
  tableros no válidos.

Las secuencias van de una solución a otra al azar sin repetir parejas, y
todos los caminos juntos forman un grafo (`data/salto_bits_N*.txt`).

- **Los caminos por bits son siempre los más cortos posibles.** Ningún camino
  necesitó saltos de más por bloqueos, y la distancia del emparejamiento es
  una cota inferior de cualquier camino. En N=5 y 6 coincide exactamente con
  la búsqueda en anchura: 21 pares a 4 y 24 a 6; 4 a 6 y 2 a 8.
- **Muy pocos tableros.** En N=5, los 45 caminos usan solo 172 tableros
  intermedios, frente a unos 34.000 explorando.

| N | parejas | longitud de los caminos | intermedios distintos | compartidos por ≥2 caminos | pasan por otra solución |
|---|---|---|---|---|---|
| 5 | 45 | 4 o 6 | 172 | 11 | 0 |
| 6 | 6 | 6 o 8 | 34 | 0 | 0 |
| 7 | 780 | 3–12 (mediana 8) | 4.883 | 477 (el más usado, por 8 caminos) | 1 |
| 8 | 4.186 | 2–16 (mediana 9–10) | 28.479 | 3.106 (el más usado, por 17) | 7 |

- **Los tableros donde se cruzan los caminos están "menos rotos".** En N=8,
  los intermedios en general tienen sobre todo 4–5 pares de reinas
  atacándose; los compartidos por ≥2 caminos, sobre todo 2–3. Los cruces son
  casi soluciones.
- **Secuencias al azar por todas las soluciones** (20.000):
  - N=5: mediana 46 saltos (el mínimo posible es 36).
  - N=6: mediana 20 (mínimo 18).
  - N=7: mediana 320.
  - N=8: mediana 845.

### 14. Un representante por grupo de simetría (`orbitas.c`)

Las 8 simetrías del tablero conservan los saltos de rey. Por eso se puede
trabajar con un representante por grupo: la distancia entre dos grupos es la
mínima entre el representante de uno y las 8 versiones del otro. En el
recorrido, la orientación del miembro elegido forma parte del estado.
Salidas en `data/orbitas_N*.txt`.

| N | soluciones → grupos | parejas | tableros intermedios (caminos por bits) | recorrido mínimo | secuencias al azar (mediana) |
|---|---|---|---|---|---|
| 5 | 10 → 2 | 45 → 1 | 172 → 3 | 36 → 4 | 46 → 4 |
| 6 | 4 → 1 | 6 → 0 | 34 → 0 | — | — |
| 7 | 40 → 6 | 780 → 15 | 4.883 → 68 | → 23 | 320 → 28 |
| 8 | 92 → 12 | 4.186 → 66 | 28.479 → 359 | → 49 | 845 → 74 |

Exploración con memoria (300 repeticiones), parando con **un miembro de
cada grupo** en lugar de con todas las soluciones, en N=5:

- en profundidad, la mediana baja de 47.673 tableros (90 %) a 7.934 (15 %);
- en anchura, de 34.621 (65 %) a 5.685 (11 %).

En N=6 hay un solo grupo, así que la solución inicial ya basta.

- El árbol se reduce unas 6 veces en N=5. En N=7 y 8, los tableros que hacen
  falta para conectar todos los grupos se reducen unas 70–80 veces.
- Hay grupos muy cercanos: en N=8, los grupos 0 y 8 están a solo 2 saltos.

**Trayectorias reproducibles entre grupos.** El representante canónico de
cada grupo es el miembro de menor número de bits, y los grupos se numeran por
ese número. Se calculan tres trayectorias: el árbol mínimo que los conecta,
la regla "al grupo más cercano no visitado" empezando por el grupo 0, y el
recorrido más corto.

- **El salto crece con las reinas que cambian de sitio.** Nunca es menor que
  ese número y casi siempre lo supera en 0–3. En N=8, 2 reinas dan 2 saltos;
  4 reinas, 4–8; 7 reinas, 7–10.
- **N=7:** el árbol mínimo es una **cadena** (1–0–2–3–4–5, saltos 6, 5, 3, 4,
  5; 23 en total). Como es una cadena, ya es el recorrido más corto, y la
  regla del más cercano también lo encuentra (23). Solo 2 órdenes de grupos
  (más sus inversos) alcanzan el óptimo.
- **N=8:** el árbol mínimo tiene dos centros (los grupos 3 y 6, con 4 ramas
  cada uno), diámetro 7 y 47 saltos. El recorrido más corto suma 49 (8 0 1 7
  3 4 11 5 9 10 6 2, saltos 2 6 5 4 3 3 5 6 5 4 6). La regla del más cercano
  da 58, y las secuencias al azar, 74 de mediana.
- En la secuencia de saltos no se ve ningún patrón regular (ni constante,
  ni creciente, ni periódico). Lo que la ordena es cuántas reinas cambian en
  cada tramo.

### 15. ¿Hay una "dirección" predecible? Moldes y saltos iguales (`moldes.c`)

Dos pruebas para el criterio "saltos parecidos y transformación
predecible". Los tableros se guardan en 256 bits, hasta N=16.

1. **Molde desplazado.** El cambio entre dos soluciones s y t es un molde
   (quitar = s AND NOT t, poner = t AND NOT s). Se aplica de nuevo, desplazado
   un vector fijo v en cada paso (24 vectores pequeños, con y sin dar la
   vuelta). Una cadena de 3 o más soluciones sería una dirección predecible.
2. **Saltos iguales.** Para cada tamaño de salto d, se conectan las soluciones
   a distancia exactamente d y se busca la cadena más larga, frente a un grafo
   al azar con los mismos grados. Dentro de la cadena se mira si el molde se
   repite desplazado.

Resultados (`data/moldes_N*.txt`):

| N | molde desplazado (≤4 reinas cambiadas) | saltos iguales: real frente a azar |
|---|---|---|
| 7 | 0 cadenas de 3 | igual o más cortas que al azar |
| 8 | 0 | ídem |
| 9 | 0 | ídem (d=5: 15 frente a 32) |
| 10 | 0 | ídem |
| 11 | 0 | ídem (d=5: 25 frente a 43) |
| 14, desde las 15.008 casi toroidales | 0 | — |

- En **ninguna** cadena de salto constante se repite el molde desplazado (0
  tramos en todas).
- Hay cadenas larguísimas con salto constante (hasta 1.495 soluciones en N=11
  con d=10), pero un grafo al azar con los mismos grados da lo mismo: es
  densidad de conexiones, no patrón.
- **Control positivo:** permitiendo 7 reinas en N=7, el detector encuentra 616
  cadenas de 7 soluciones dando la vuelta (subir todas las filas: las
  toroidales). El falso, 0. El método funciona; simplemente no hay moldes
  pequeños.

**Conclusión.** No se ha encontrado una transformación pequeña y repetible que
lleve de una solución a la siguiente. Las únicas "direcciones" que existen
mueven todas las reinas a la vez: las simetrías, el giro de filas en el toro
y su versión casi toroidal (secciones 9–10).

### 16. No toroidales: vistazo lineal (secciones 1–2)

- Todas las soluciones cumplen Σ x_i = Σ i, porque son permutaciones. Es una
  restricción trivial y no aporta nada.
- **N=7 es la excepción:** las 12 no toroidales tienen dimensión afín 5. Eso
  da un subárbol de profundidad 2 con las 12. Además de Σx, cumplen
  (0,1,−1,2,−1,1,0)·x ≡ 6 (mod 7). Falta ver si es estructura o efecto de
  ser solo 12.
- **N=11 y 13:** dimensión 10 de 11 y 12 de 13, sin ningún funcional lineal
  extra. Para ellas harán falta niveles no lineales (cuadráticos) o varios
  grupos en subárboles distintos. En N=7, el mejor funcional lineal no
  constante deja 8 de las 12 en un mismo subárbol.
