# Swap Queens: de una solución normal a las espirales

Pregunta: ¿intercambiando X parejas de celdas se llega desde una solución normal
a una toroidal de una o dos espirales?

Cada pareja (reina, celda vacía) mueve una reina, así que X es el número de filas
en que difieren las dos permutaciones. X = 1 es imposible. `cerca.c` calcula el X
mínimo de cada solución a cada familia. Compilar y usar: ver la cabecera del archivo.

Objetivos: 1 espiral = N(N−3); 2 espirales = 2.366 / 7.514 / 14.440 (N = 13 / 17 / 19),
iguales a la fórmula. N = 5, 7, 11 no tienen dos espirales.

Datos: N ≤ 17 exhaustivo; N = 19, 1 de cada 1.000 soluciones. Dos nulos: permutaciones
al azar, y soluciones reales contra objetivos con filas y columnas barajadas. Dan lo mismo.

| N | X medio real | X medio azar | real X ≤ 4 | azar X ≤ 4 |
|---|---|---|---|---|
| 13 | 5,74 | 7,17 | 14,2 % | 0,26 % |
| 17 | 9,75 | 10,67 | 0,069 % | 0,000025 % (2 casos en 8 M) |
| 19 | 11,45 | 12,31 | 0,0039 % | 0 en 2 M |

X = cualquiera de las dos familias. Solo soluciones no toroidales.

- Lo típico: X/N = 0,44, 0,57, 0,60; la ventaja sobre el azar baja de 1,4 a 0,9 intercambios.
- La cola: hay soluciones a X = 2 en todos los N probados, con exceso grande frente al azar.
- Toda espiral tiene una normal cerca: X ≤ 4 (una), X ≤ 5 (dos, N = 13), X ≤ 4 (dos, N = 17).
- Entre toroidales X = 2 y 3 no aparecen; el mínimo es 4, como ya se sabía.

## La misma receta sobre el tablero (`receta.c`)

Ahora el intercambio es fijo: unas celdas concretas del tablero, y se mira si llevan
espirales a espirales. Partida y llegada son siempre toroidales de una o dos espirales.
Es la misma medida en el tablero cuadrado y en el circular: filas = anillos, columnas = ángulos.

X posibles entre espirales (exhaustivo):

| N | una ↔ dos | dos ↔ dos, mínimo | una ↔ una |
|---|---|---|---|
| 13 | 4, 6, 8, 11, 12, 13 | 4 | 12, 13 |
| 17 | 4, 8, 12, 15, 16, 17 | 4 | 16, 17 |
| 19 | 6, 9, 12, 17, 18, 19 | 6 | 18, 19 |

Mismo molde (U xor V) repetido en varios pares: máximo 4 pares en N = 13 y 19, 8 en N = 17.
Con permutaciones al azar del mismo tamaño nunca se repite.

**Regla observada.** Las recetas útiles reflejan c → s − c en un conjunto de filas con un
centro. Probadas todas las S = −S alrededor de una fila, aplicadas a toda la familia:
sirven exactamente las uniones de clases de un subgrupo μ:

| N | subgrupo | recetas útiles | de | mejor receta parcial |
|---|---|---|---|---|
| 13 | μ4 o μ6 | 9 | 63 | 8 de 2.496 |
| 17 | μ4 | 15 | 255 | 24 de 7.752 |
| 19 | μ6 | 7 | 511 | 16 de 14.744 |

Reflejar todas las filas menos la del centro mueve solo las que pasan por el centro
(192, 456 y 776); con esa fila incluida es la simetría global. Ejemplo N = 17: centro en la
fila 11, filas 11 ± {6, 7, 10, 11}, 8 pares con 4 reinas movidas.

Lectura: una espiral queda partida en dos (a y −a) al invertir una clase de filas. Encaja con
la regla ya probada de dos espirales: S es unión de clases de H = ⟨−1, (1+a)/(1−a)⟩. μ4 sale
con a = ±i, por eso la receta vale para pendientes concretas, no para todas.

## Permutar todo el tablero (`tablero.c`, `forma.c`)

La idea original: reordenar libremente las N² celdas, sin límite de intercambios, y mirar si un
conjunto de soluciones no toroidales acaba con forma de espirales. Solo cuenta cómo se solapan:
existe la permutación sii cada intersección (átomo) mide igual en origen y destino.

**Particiones (`tablero.c`, exhaustivo).** N soluciones disjuntas = una pendiente completa.
N = 7: 4 clases, todas de rectas; N = 11: 8. Sin toroidales: 0 en ambos. Ortogonalidad y Z_N
de las rectas salen como deben (control). N = 13 no terminó en 3 min en ningún modo: sin resolver.

**Cuántas caben a la vez (`forma.c`, voraz: cotas inferiores).**

| N | destino | no toroidales: mejor / media | azar: mejor / media | rectas reales: mejor / total |
|---|---|---|---|---|
| 7 | una | 4 / 3,63 | 5 / 3,84 | 28 / 28 |
| 11 | una | 11 / 8,71 | 11 / 8,96 | 25 / 88 |
| 13 | una | 12 / 9,40 | 12 / 9,64 | 20 / 130 |
| 13 | una o dos | 13 / 11,90 | 14 / 12,07 | — |

Las soluciones no toroidales no forman más geometría que permutaciones al azar. El voraz no
recupera ni las rectas reales en N ≥ 11, así que estos números no son máximos.

## Girar filas hasta llevar S a T (`figura.c`)

Grupo 1: una y dos espirales. Grupo 2: el resto. Se eligen T y S al azar, cada fila gira
T[r] − S[r] y ese tablero se aplica a toda U. Acierto: U forma una figura sobre una o dos rectas
mod N, sea o no solución. Fuera: U = S, S girada entera, y lado menor < 4 (dos puntos siempre son
recta). Datos en `data/figura.txt`. N = 13: 200.000 intentos, controles 100.000.

Aciertos por intento:

| N | T | real | U azar | tablero azar | T azar |
|---|---|---|---|---|---|
| 13 | una | 0,533 | 0,025 | 0,056 | 0,011 |
| 13 | dos | 0,063 | 0,013 | 0,017 | 0,011 |
| 11 | una | 0,393 | 0,080 | — | 0,048 |

Figura que además es solución toroidal exacta, N = 13: una 0,0285, dos 0,0061 por intento;
controles, 1, 2 y 0 casos en 100.000.

N = 17, con 1 de cada 96 soluciones: 1 acierto en 660 intentos (una), 3 en 19.340 (dos), 0 en los
controles. Son muy pocos para concluir.

Límite: con T recta, f = T + (U − S) cabe en dos rectas sii U − S cabe; no depende de T, mide la
relación entre U y S. Con T de dos espirales no se ha medido cuánto depende de T.

## ¿Hay listas de rotaciones mejores? (`rotaciones.c`)

Censo exacto, no muestreo: todos los pares (U del grupo 2, V del grupo 1) dan todas las listas que
aciertan. Acierto = U + lista es exactamente una solución del grupo 1. Datos en `data/rotaciones.txt`.

| N | listas con aciertos | máximo | con ≥ 2 | máximo U azar / V azar |
|---|---|---|---|---|
| 11 | 17.932 | 2 | 84 | 1 / 1 |
| 13 | 10.725.956 | 8 | 14.986 | 2 / 2 |
| 17, 1 de cada 192 U | 226.410.624 | 2 | 672 | 1 / 1 |

En N = 13, el 99,9 % de las listas acierta una vez. Las 30 con 6, 7 u 8 aciertos son casi constantes:
se apartan del giro común en 2 o 3 filas. Es decir, mueven 2 o 3 reinas y no giran el tablero.
Las de 5 aciertos con 11 filas apartadas (38) no son excepción en las 6 miradas: son −r + c o r + c
con 3 filas cambiadas. Quedan 24 listas de 4 aciertos con 7 filas apartadas sin explicar; vistas 1.

## Saltos entre soluciones sin el grupo 1 (`saltos.c`)

Lista d: cada fila gira d[r]. Salto: U y U + d son soluciones y ninguna es de una o dos espirales.
Una lista al azar casi nunca acierta, así que se recorren todos los pares {U, W}: W − U son todas las
listas que aciertan. d y −d cuentan juntas. Datos en `data/saltos.txt`, N = 8…13.

El espejo de columnas duplica casi todo: U → W y su reflejo usan la misma lista, así que en las
soluciones reales 2 saltos equivalen a un par. Por eso las cuentas pares no se comparan tal cual con
el azar. En N = 13 G incluye toroidales de 3+ espirales: aportan 2.028 saltos a cada giro constante y
la familia de 26 saltos (74.474 listas), que es su propia traslación, no estructura nueva.

Mejores listas:

| N | giro 1 | giro 2 | giro 3 | mejor que no es recta (2 filas) | máx. azar |
|---|---|---|---|---|---|
| 11 | 336 | 50 | — | 20 | 3 |
| 12 | 2.640 | 612 | 184 | 50 | 3 |
| 13 | 15.178 | 4.764 | 2.534 | 232 | 3 |

- Gana siempre el giro constante, y cuanto menor, mejor.
- Las siguientes cambian solo 2 filas con d y N − d: intercambiar dos reinas entre dos filas.
  Salen sobre todo con filas de los bordes.
- Listas de verdad globales (≥ 7 filas fuera de toda recta): N = 12, máximo 14 saltos, unos 7 pares.
  Sin comprobar si las listas capicúa duplican más.

## Sin toroidales ni simetrías: listas y caminos (`cadenas.c`)

G = soluciones no toroidales. Cada par y cada lista se cuentan una vez bajo espejo de columnas, espejo
de filas y giro de 180°. Censo exacto en `data/cadenas/`.

| N | giro 1 | giro 2 | giro 3 | mejor de 2 filas | máx. con ≥ 5 filas fuera de recta | azar máx. |
|---|---|---|---|---|---|---|
| 12 | 660 | 155 | 46 | 25 | 9 | 2 |
| 13 | 3.317 | 690 | 130 | 86 | 17 | 2 |

Las listas con ≥ 5 filas fuera de recta son sobre todo locales: la mejor de N = 13,
`0 0 1 0 0 9 3 10 4 0 0 12 0`, mueve solo 6 filas.

**Misma lista repetida (exacto), soluciones seguidas:**

| N | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|
| giro constante | 3 | 5 | 4 | 5 | 6 | 6 |
| lista no constante | — | — | 4 | — | 4 | 4 |

Coincide con `camino_queens/giros.c`. En azar hay cadenas sueltas de 3 (N = 10, 13).

**Listas distintas en cada salto (`data/posa/`, Pósa, cota inferior):** arista si la lista aparece en
K órbitas o más y tiene ≥ 5 filas fuera de recta. Con cualquier lista el grafo es completo.

| N | K | nodos del mayor componente | camino |
|---|---|---|---|
| 12 | 2 | 14.154 | 14.154 |
| 12 | 3 | 9.628 | 901 |
| 12 | 4 | 2.980 | 460 |
| 12 | 6 | 21 | 17 |
| 13 | 3 | 64.050 | 6.164 |
| 13 | 4 | 28.986 | 665 |
| 13 | 6 | 812 | 22 |

Con K = 2 el camino no significa nada: el azar da 62.829 de 63.232 en N = 13. Con K ≥ 3 el azar casi
no tiene grafo, y estos caminos son cotas inferiores.

## Islas por niveles de reglas (`islas.c`)

Nodos: no toroidales, una por grupo de simetría (8). Nivel k: se unen dos soluciones si difieren en k
filas o menos o son giro una de otra. Datos en `data/islas.txt`, N = 8, 10, 12, 13.

N = 13, 8.661 grupos:

| k | islas real | mayor real | profundidad | mayor azar |
|---|---|---|---|---|
| 0 (solo giros) | 7.141 | 0,1 % | 4 | 0,0 % |
| 2 | 4.554 | 0,6 % | 15 | 0,0 % |
| 3 | 1.464 | 73,7 % | 29 | 0,0 % |
| 4 | 72 | 99,1 % | 10 | 0,0 % |
| 5 | 2 | 100 % | 5 | 0,4 % |
| 6 | 1 | 100 % | 4 | 99,7 % |

Con giro libre antes de cambiar k filas, el salto es en k = 2 (57 %), azar en k = 4–5. N = 12: mismo
salto en k = 3 (63 %). El salto ocurre cuando hay tantas aristas como nodos, como en cualquier grafo
al azar: lo propio de las soluciones es tener más vecinos cercanos, no un mecanismo de unión.

## Dentro de las islas de nivel 2 (`dentro.c`)

Miembros alineados recorriendo la isla. Núcleo = filas con la misma reina en todos. W = pares de
costura. X = reinas hasta la toroidal más cercana. Eje = parte de la isla a X ≤ 4 de una misma
toroidal. N = 13, medias de `data/dentro.txt`:

| tamaño | islas | núcleo | bucles | W | X | eje |
|---|---|---|---|---|---|---|
| 1 | 3.114 | 13,0 | 0 | 4,99 | 5,42 | — |
| 2 | 762 | 11,8 | 0 | 4,85 | 5,34 | — |
| 3–4 | 403 | 7,5 | 0,2 | 4,67 | 5,27 | 21 % |
| 5–9 | 199 | 2,5 | 0,7 | 4,63 | 5,25 | 18 % |
| 10–29 | 65 | 0,5 | 2,4 | 4,49 | 5,08 | 13 % |
| 30+ | 11 | 0,3 | 8,5 | 4,41 | 4,76 | 8 % |

Son casi árboles, pero sin raíz ni núcleo: desde 10 miembros no queda ninguna fila común. El eje
máximo es 38 %. N = 12 no tiene toroidales y da el mismo patrón, así que las islas no son satélites
de las espirales. W y X bajan poco con el tamaño. Sin nulo para el núcleo.

## Islas que son copias unas de otras (`familias.c`)

Etiqueta de cada conexión: giro c; vector = intercambio con separación (filas, columnas); lugar = el
mismo rectángulo del tablero salvo simetría. Huella: forma + etiquetas (Weisfeiler-Lehman, no prueba
isomorfismo). Solo islas de 3 o más, sin las hechas solo de giros. Nulo: mismas islas con las
etiquetas barajadas, media de 20. Datos en `data/familias_interc.txt`.

| N | islas | repetidas, vector (nulo) | repetidas, lugar (nulo) | mayor isla repetida |
|---|---|---|---|---|
| 12 | 114 | 11 (8,3) | 0 (1,4) | 4 |
| 13 | 611 | 114 (89,7) | 47 (19,8) | 4 |
| 14 | 3.162 | 813 (679,4) | 276 (215,4) | 6, 2 copias |

Solo por forma se repite casi todo (84 % en N = 14), pero son formas pequeñas. Con etiquetas, el
exceso sobre el nulo es pequeño y las copias son de 2 o 3. Las mejores son tipo g1, (a,b), g1, (a,b):
un intercambio y un giro aplicados en los dos órdenes.
