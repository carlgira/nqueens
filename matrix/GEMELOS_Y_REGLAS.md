# Grupos gemelos y reglas para limitar caminos

Concentra los experimentos de `matrix/` sobre dos preguntas: cuándo salen dos grupos grandes gemelos y cuánto
reducen los caminos posibles las reglas probadas. Cifras replicadas o medidas el 2026-11-01.

## Resumen

- Los gemelos no son un solo fenómeno: hay **tres familias** con causas distintas.
- Solo la familia de **paridad** es general y está explicada. Las otras dos son efectos de N pequeño.
- Las reglas de **aristas** reducen mucho los candidatos. Ninguna regla de **elección** probada resume un camino
  completo: la que llega al final ocupa más que señalar el camino directamente.

## Familias de gemelos

| Familia | Dónde sale | Qué los separa | Quién los intercambia |
|---|---|---|---|
| Paridad | Solo ciclos de 3, solo dobles o ambos | Invariante exacto: paridad de q | N mod 4 = 2, 3: reflejos. N mod 4 = 0, 1: nadie |
| Involución de N=8 | T involutiva, solo N=8 | Desconocido; 0 invariantes lineales mod 2 | Transposición |
| Cayley ≤ 2 con N pequeño | N=8 y N=10 | Casi paridad, unida por pocos swaps | Reflejos |

### 1. Paridad

Ciclos de 3 y dobles conservan la paridad de q, de modo que cada bloque es de una sola paridad.

| N | N mod 4 | Ciclo3+doble: bloques | Iguales | Reflejo los intercambia |
|---|---|---|---|---|
| 12 | 0 | 5.532 / 5.208 | No | No |
| 13 | 1 | 34.396 / 29.832 | No | No |
| 14 | 2 | 165.768 / 165.768 | Sí | Sí |
| 15 | 3 | 1.081.724 / 1.081.724 | Sí | Sí |
| 16 | 0 | 7.188.376 / 7.179.944 | No | No |

Explicación exacta: invertir las filas (o las columnas) tiene signo (−1)^⌊N/2⌋ y transponer conserva la paridad.
Con N mod 4 = 2 o 3 un reflejo cambia la paridad y por eso los bloques son espejos exactos. Con N mod 4 = 0 o 1 las
8 imágenes tienen la misma paridad, los bloques no se relacionan y difieren de tamaño.

Con los tres movimientos juntos desaparecen como gemelos: desde N=11 hay un gigante único, y sus dos paridades quedan
casi equilibradas (N=16: 7.272.444 pares, 7.280.980 impares). En N=9 también es único, pero con reparto 160/104.

### 2. Involución de N=8

"T es involución" da 46 + 46 nodos. Es la familia más llamativa y la menos general:

- Cada componente tiene 22 pares y 24 impares: no es paridad.
- Reflejo de filas y de columnas se quedan dentro. Las 4 imágenes con transposición pasan a la otra.
- `analyze.py` no encuentra ningún invariante lineal mod 2.
- **Desaparece ya en N=9**, no en N=16 como sugería la guía: de N=9 a N=14 hay una sola componente con el 100 %.
  El grado de involución crece de 4,0 a 36,2, así que el grafo se llena.

### 3. Cayley ≤ 2 con N pequeño

N=8 da 38 + 38 y N=10 da 226 + 226. En N=10 un grupo tiene 166 pares y 60 impares, y el otro al revés. Son los bloques
de paridad (162 + 162 con ciclo3+doble) unidos por pocos swaps. Los reflejos los intercambian. No se repite después:
N=9 y N=11 ya tienen un componente mayor sin gemelo y N=12 llega al 85,7 %.

## Reglas y cuánto reducen

### Reglas sobre aristas (qué movimientos existen)

| N=16, por solución | Trabajo real | Resultado |
|---|---|---|
| Fuerza bruta | Prueba 6.700 candidatos | 5,4 vecinos |
| Filtros A y B | También miran los 6.700; pasan 4.727 y 9,0 | 5,4 vecinos |
| Generador guiado | Tabla de unas 750 operaciones y 96 candidatos | 5,4 vecinos |

El único que evita mirar los 6.700 es el generador. Los 5,4 vecinos no son un coste: son la respuesta, que no se puede
bajar sin perder vecinos. Las reglas abaratan buscarlos, pero no deciden cuál tomar. Las reglas fijas (sin D4, sin
toroidal-toroidal) casi no influyen: quitan 3.028 aristas de 39,9 millones en N=16.

Comparación con el backtracking con máscaras, por solución obtenida en N=16:

| Método | Coste por solución | Memoria extra | Cobertura |
|---|---|---|---|
| Backtracking | 77 nodos (433 ns) | Ninguna | 100 % |
| Grafo con generador | Unas 850 operaciones, más visitados | Conjunto de visitados | 98,5 % |
| `nq_cayley` (fuerza bruta) | 6.700 candidatos (70 µs de núcleo, con su contención) | Lista completa | 98,5 % |

El generador crece aproximadamente como N² en lo medido (N=8 a 16); el backtracking se mantiene casi plano (60 a 77 nodos
de N=12 a 16). Enumerarlas todas sigue siendo Q(N) veces ese coste por cualquier vía, y una sola solución se construye
en O(N), así que el grafo no mejora ninguno de los dos casos. El generador no está cronometrado aparte y una operación
no equivale exactamente a un nodo.

Actualizar la tabla al moverse (`nq_incremental.c`, paseo real de 100.000 pasos en N=16, 0 diferencias con reconstruir):
pasa de 496 a 245 operaciones, pero tarda lo mismo (unos 530 ns), porque cada operación incremental es más cara. Un
recorrido DFS hace 2 movimientos por solución (ida y vuelta), lo que anula esa mitad. Reconstruir cuesta N²; actualizar
crece casi lineal en lo medido (110 en N=8, 245 en N=16).

Mejor opción medida: **no guardar tabla**. Basta la ocupación de las diagonales, con unas 13 escrituras por movimiento, y
reclasificar las 240 jugadas: 253 operaciones y 394 ns en N=16, un 27 % menos de tiempo que reconstruir. Sigue siendo N²
por las jugadas. En DFS la vuelta solo escribiría diagonales, sin reclasificar, así que no se duplicaría (deducido, no medido).

### Reglas sobre el camino (N=8, involución)

| Regla | Resultado |
|---|---|
| Sin regla, 92 soluciones | 92! ≈ 10^142 órdenes |
| Una componente de 46, todas las T (32) | 8.048.440 caminos hamiltonianos |
| Solo las 31 T más aplicables | 46.368 |
| Solo las 30 más aplicables | 0 |
| Cociente por reflejo de filas (23 clases) | 3.236 secuencias, 36.368 con etiqueta T |
| Cociente con el mínimo de 13 T | 4 con etiqueta T |
| Palabra periódica de longitud 2, 3, 4 | 4, 6, 8 de 46 nodos |
| Lista de prioridad fija | Llega a 46 desde su mejor inicio; media 42,1 |

Lectura:

- **Quitar una sola T cuesta mucho.** De 32 a 31 T quedan 174 veces menos caminos; con 30 no queda ninguno. Depende del
  desempate entre las T con igual aplicabilidad, así que el umbral exacto no es universal.
- **El cociente no da un camino de 46.** Sus caminos recorren 23 nodos y el otro lado es la imagen por el reflejo, con la
  misma palabra T. Para unirlos haría falta una arista de x a su reflejo, y no existe ninguna. Todos los conteos están
  dirigidos, así que cada camino se cuenta dos veces.
- **Las palabras periódicas fallan.** Cubren como mucho 8 de 46, y dentro de un camino la subpalabra repetida más larga es
  de 3 pasos de 22. En longitud 3 y 4 solo se probaron las 30 y 18 T más aplicables.
- **La lista de prioridad no comprime.** Elegir uno de 8.048.440 caminos cuesta 23 bits; una lista de 32 T cuesta hasta
  118. La regla guarda más información que el resultado: es una búsqueda ajustada, no una ley.

## Límite de fondo

Un camino hamiltoniano exige como mucho 2 nodos de grado 1. El gigante tiene 2.308 en N=12, 34.324 en N=14 y 631.424 en
N=16, de modo que en Cayley ≤ 2 no existe. Cualquier regla de recorrido debe permitir repetir nodos o saltos mayores.

## Híbrido: backtracking y grafo

`nq_hibrido.c` mide dónde caen los vecinos en el orden del backtracking y simula una poda: al llegar a un prefijo de D
filas solo se entra si alguna solución ya hallada predice una solución dentro. El coste cuenta nodos y no incluye predecir,
y solo las soluciones anteriores pueden predecir, como pasaría en una pasada real.

Los vecinos están lejos en el árbol. En N=16 el 26,1 % de las aristas cambia ya la primera fila y el 57,8 % alguna de las
tres primeras; la distancia media es el 10,9 % de todo el orden. Estar cerca en el grafo no es estar cerca en el árbol.

| N=16, corte D | Nodos visitados | Soluciones halladas |
|---|---|---|
| 4 | 99,9 % | 99,99 % |
| 6 | 39,9 % | 40,2 % |
| 8 | 20,6 % | 34,5 % |
| 10 | 18,5 % | 20,5 % |
| 12 | 54,6 % | 8,0 % |

El mejor punto rinde 1,7 veces más que cortar a ciegas y pierde dos tercios. En N=14 y N=12, con D alto, la predicción se
apaga casi por completo. Además, predecir cuesta cientos de operaciones por solución frente a 77 nodos, así que tampoco
compensaría aunque recortase perfecto. El motivo de fondo: el grafo dice dónde **hay** soluciones, y podar exige saber
dónde **no hay**.

Lo que sí se puede recortar sin perder soluciones:

- **Nodos muertos:** el 89,8 % de los nodos de N=16 no lleva a ninguna solución (7,9 vivos por solución). Están en las filas
  8 a 15, con 59 a 95 % de muertos. Es el techo: como mucho 10 veces menos nodos.
- **Fusión de estados** (`nq_estados.c`): dos prefijos con mismas columnas y diagonales tienen las mismas continuaciones.
  Quedan 82,8 %, 74,9 %, 69,6 % y 63,4 % de los nodos en N=12, 14, 15 y 16. Mejora con N, pero con tabla hash tarda 97 s
  frente a 6,4 s, usa 194 millones de estados en el pico y solo sirve para contar, no para listar.

## De una solución a la onda

Sobre un componente C del grafo, L = D − A cumple L|u⟩ = 0 con |u⟩ = (1/√|C|) Σ|s⟩: la onda uniforme es exactamente su estado
de energía mínima. `nq_onda.c` calcula el gap de L y evoluciona la distribución de forma exacta desde la solución construida,
con un paseo de distribución estacionaria uniforme (P = I − L/(dmax+1), paso perezoso).

| N | |C| | % de Q | λ₂ de L | Nodos del modo lento | TV < 1/4 | TV < 1/100 |
|---|---|---|---|---|---|---|
| 12 | 12.172 | 85,72 | 0,0278 | 88 | ≤ 2.048 | ≤ 4.096 |
| 13 | 68.044 | 92,31 | 0,0254 | 22 | ≤ 512 | ≤ 1.024 |
| 14 | 346.816 | 94,86 | 0,0213 | 57 | ≤ 512 | ≤ 2.048 |
| 15 | 2.216.072 | 97,23 | 0,0356 | 31 | ≤ 512 | ≤ 1.024 |
| 16 | 14.553.424 | 98,52 | 0,0361 | 13 | ≤ 512 | ≤ 2.048 |

Las columnas de TV son pasos perezosos, medidos en potencias de 2 (cotas superiores); hay que dividir entre unas 10 para
tener movimientos reales en N=16. El gap no cae aunque |C| se multiplique por 1.000, y el modo lento es una cola local de
pocos nodos, no un cuello global (en N=9 y 11 sí es global). La paridad no es ese modo: solape del orden de 10⁻³ o menor.

Qué se recupera desde una solución:

- **Las probabilidades, sí.** En N=16, a 2.048 pasos TV = 0,00026 y el mapa de ocupación por celda difiere un 0,06 %.
- **La onda coherente, no por esta vía.** El paseo da la mezcla ρ = I/|C|, con la misma diagonal que |u⟩⟨u| pero sin
  elementos fuera de la diagonal. El tiempo imaginario e^(−tL) sí llega (fidelidad 0,957 en N=16), pero no es unitario.
- Desde |s⟩ el solape con |u⟩ es 1/√|C| (1/3.815 en N=16). Sin un camino mejor, la amplificación costaría √|C|, exponencial.
- Solo se alcanza el componente de la solución. L tiene un estado fundamental por componente (196.625 en N=16).

Límite: una sola salida (la construida) y N ≤ 16; que el gap se mantenga con N mayor no está probado.

## Atractores y onda real sobre el grafo

La igualdad de probabilidades de la sección anterior está **impuesta**, no descubierta: L incluye el grado como energía de
sitio y por eso sale plana. Sin ese término aparecen atractores (`nq_campo.c`, componente mayor).

| N=16 | Solución más probable | Soluciones efectivas |
|---|---|---|
| Término de grado incluido (L) | x1 | 100 % |
| Paseo natural (proporcional al grado) | x10,2 | 78,6 % |
| Mínima energía de −A (solo saltos) | x377.181 | 125 de 14,5 millones |

Con solo saltos, el 0,1 % superior acumula el 100 % en N=16 (36,7 % en N=12, 98,6 % en N=14). La solución de mayor grado suele
ser muy ordenada: en N=16 es 4 8 12 16 3 7 11… (paso +4 mod 17) y en N=12 es 5 10 2 7… (paso +5 mod 13). Eso coincide
con el pico de −A en N=10 y 12 (dos lineales suman el 28 % y el 24 %), pero no de N=13 a 16: en N=16 el pico tiene grado 39
y las 14 lineales apenas pesan.

Onda unitaria |ψ(t)⟩ = e^(−iLt)|s⟩ desde una solución:

- Salta por el grafo y se extiende: N=16 llega a t=64 con el 37 % de soluciones efectivas, y N=14 se estabiliza en 0,30 a
  0,35 desde t≈256. Mapa de celdas a un 3 % del exacto.
- Queda como un **moteado**, no plana: TV estable en 0,42 a 0,45. Hay coherencia sobre todo el componente.
- Su solape con la onda plana es **exactamente constante**, 1/|C|, porque |u⟩ es autoestado de L. La simulación lo confirma
  (6,871 × 10⁻⁸ en N=16 en todos los tiempos). Evolucionar con L nunca aplana la onda.

En t=4096 la norma deriva 10⁻⁴: esa fila es menos fiable.

La partícula-swap como operador: A = Π (Σ T) Π, donde T recorre los movimientos y Π proyecta sobre soluciones. Choca con 6.700
candidatos y sobreviven 5,4.

## Subgrafos: cadenas, cobertura y superstring

`nq_subgrafos.c`. Regla = palabra periódica de T con etiquetas de columna exactas, aplicada una y otra vez. Exhaustivo hasta
periodo 3 en N≤14 y periodo 2 en N=15 y 16.

| N | Mayor cadena p=1 | p=2 | p=3 | Mayor ciclo | Mejor regla actúa en | 100 reglas cubren |
|---|---|---|---|---|---|---|
| 12 | 2 | 4 | 7 | 6 | 0,27 % | 8,3 % |
| 13 | 2 | 6 | 6 | 6 | 0,10 % | 5,0 % |
| 14 | 2 | 6 | 9 | 6 | 0,05 % | 3,1 % |
| 16 | 3 | 8 | sin medir | 6 | sin medir | sin medir |

Las reglas fijas dan piezas diminutas y cada vez cubren menos: T tiene orden 2 o 3, y las cadenas largas son rarísimas (N=16,
192 de 478 millones llegan a 8). La cobertura usa las 400 reglas más frecuentes de periodo 2 (voraz), no todas.

Sin regla sí salen piezas largas. Cubrir por caminos (voraz, con Warnsdorff):

| N | Soluciones con vecino | Caminos | Longitud media | Mayor | Mínimo posible por hojas |
|---|---|---|---|---|---|
| 12 | 13.088 | 2.224 | 5,9 | 38 | 1.514 |
| 14 | 352.040 | 31.690 | 11,1 | 117 | 19.270 |
| 15 | 2.231.280 | 132.209 | 16,9 | 211 | 80.568 |
| 16 | 14.594.552 | 530.964 | 27,5 | 481 | 333.980 |

La media crece rápido y queda a 1,6 veces del mínimo. Pero cada paso es una elección, de modo que no hay regla corta.

Superstring voraz (solapar el final de una solución con el principio de otra):

| N | Longitud frente a concatenar | Dígitos nuevos por solución | Nodos vivos del árbol por solución |
|---|---|---|---|
| 12 | 51,7 % | 6,21 | 7,4 |
| 14 | 49,2 % | 6,88 | 7,7 |
| 16 | 47,3 % | 7,57 | 7,9 |

Comprime a la mitad, y casi igual que el árbol de backtracking, que ya comparte prefijos. En N=16 el 20 % de los empalmes
solapa 15 dígitos, es decir, son rotaciones de filas válidas.

## Función determinista que elige el swap

`nq_caos.c`. El paseo no consulta qué soluciones ya visitó; el mapa de visitados solo sirve para medir. Límite de 40·|C| pasos,
sale de la solución construida. N=16:

| Quién elige | Visita | Pasos/|C| hasta 99 % | Hasta 99,9 % |
|---|---|---|---|
| Pseudoaleatorio con semilla (o hash de solución y paso) | 99,9 % | 16,6 | 37 |
| Rotor, un contador por solución | 99,9 % | 8,6 | 13,3 |
| Logístico, 1 iteración por paso | 0,4 a 3,5 % | Nunca | Nunca |
| Rotación áurea (Sturmian) | 0,01 % | Nunca | Nunca |
| hash(s), sin estado ni reloj | 0,00 % | Nunca | Nunca |

- Hace falta bastante entropía por paso. Elegir entre d vecinos pide log₂ d bits y el logístico da 1 por iteración. En N=14, con
  k iteraciones por paso visita 97 % (k=2), 99,7 % (k=3) y 99,8 % (k=4). En N=16 se queda en 75 a 87 %, probablemente porque
  el logístico en `double` tiene periodo 4·10⁷, menor que los pasos usados; no está comprobado.
- Los paseos caóticos cambian entre ejecuciones porque el orden de vecinos depende de los hilos. Valga el orden de magnitud.
- hash(s) cae en un ciclo enseguida: una función de la solución sola tiene que repetirse al volver a un nodo.
- Sturmian funcionaba para construir soluciones (Sturmian Queens) y aquí es el peor: no hay entropía.

Para no repetir sin memoria hace falta un padre único (búsqueda inversa): cada solución calcula su padre con una función propia,
y todo forma un árbol. Cada mínimo local es un árbol aparte, y haría falta uno solo.

| Función | Mínimos locales N=12 | N=14 | N=16 | Árbol medio N=16 |
|---|---|---|---|---|
| Hash caótico | 29,8 % | 24,4 % | 19,0 % | 5,3 |
| Índice lexicográfico | 24,1 % | 17,7 % | 11,5 % | 8,7 |
| Distancia a una referencia (L1) | 23,2 % | 17,1 % | 10,7 % | 9,3 |
| Mayor grado | 9,9 % | 6,9 % | 3,8 % | 26,5 |
| Distancia BFS (con memoria) | 1 | 1 | 1 | Todo |

El caos es la peor función para esto y la estructura (grado) la mejor, aunque sigue lejos de 1.

## Abierto

- Qué separa las 46 + 46 de N=8.
- Por qué el grafo de solo swaps queda tan fragmentado (N=16: 698 frente a 2,27 millones en un grafo aleatorio con los
  mismos grados).
- Una regla de elección corta para N mayor: lo medido aquí solo cubre N=8.

## Replicar

```bash
clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -o nq_gemelos nq_gemelos.c
clang -O3 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp -o ham_componente ham_componente.c
```

```bash
./nq_gemelos 16
```

```bash
python3 build_graph.py 8 && python3 export_componente.py > g46.txt && ./ham_componente g46.txt
```

`nq_gemelos` calcula involución solo hasta N=14. `ham_componente` tarda unos 2 minutos con 12 núcleos.
