# N reinas: síntesis y cierre (julio – noviembre 2026)

Este documento resume cinco meses de experimentos sobre las N reinas: qué se
buscaba, qué quedó establecido, qué salió negativo y dónde está la frontera.
El detalle de cada línea está en el README de su carpeta.

## 1. La pregunta

La hipótesis de fondo, en sus distintas formas:

> Las soluciones de las N reinas no son un conjunto amorfo. Existe un orden
> (una onda por reina, una dinámica, una regla recursiva) que las organiza en
> grupos con un patrón común, de modo que un mecanismo simple recorra o genere
> todas, o casi todas.

Se atacó desde unas veinticinco direcciones: física (partículas, péndulos,
osciladores, fluidos), ondas y Fourier, aritmética (fracciones, polinomios,
conteo tipo ζ), geometría y topología (toro, cámaras, epiciclos, fractales),
y combinatoria (reglas, fusión, grafos de intercambios).

## 2. Veredicto

**El orden existe, pero se acaba en la familia modular.** Todas las vistas
coinciden en lo mismo:

1. **Todo lo que tiene estructura es la familia lineal/toroidal mod N** o está
   a pocas reinas de ella. Cada codificación la redescubre con otro nombre:
   espiral, estado *splay*, onda única de rejilla, rotación de entropía cero,
   nudo tórico, factor del producto tensorial, cola diofántica `(N−1)²`,
   bloque contiguo en el orden D2, núcleo de Dirichlet.
2. **Las soluciones genéricas son casi indistinguibles del azar** una vez que
   se compara contra el control correcto (permutaciones con la misma simetría).
   Forma, intercambios, cadenas, espectro, distancia: todo da "azar con un
   ligero exceso".
3. **Incluso dentro de las toroidales el orden se extingue.** Fracción con
   descripción: 100 % (p=13), ~62–74 % (17), 50 % (19), ~3 % (23), <0,01 % (29).
4. **Las dinámicas encuentran soluciones mejor que el azar, pero no las
   generan con estructura**: ninguna encadena soluciones genéricas distintas
   siguiendo una ley.

La hipótesis se cumple en islas concretas (las toroidales, los bloques de
N=16, las familias de traslación) y no como principio general. Es un resultado
negativo sólido, repetido de forma independiente, y coincide con la literatura:
no hay fórmula para todas las soluciones toroidales, solo familias
(ortomorfismos cuadráticos y ciclotómicos; OEIS A071607 marcada *hard*).

## 3. Lo que quedó exacto

Resultados con prueba o con verificación exhaustiva sin excepciones.

**Estructura de las soluciones**

- **Espirales = toroidales lineales.** `sol[j] = (dir·2d·(j+1) − i) mod N`,
  con `(N−3)·N` soluciones para N primo. Coincide con el conteo toroidal en
  N=5, 7, 11; N=13 es el primer primo con toroidales no lineales (130 de 4524).
- **Producto tensorial** (`fractal_queens/`): `σ ⊗ τ` es solución para toda σ
  válida **si y solo si τ es toroidal**. Da torres autosemejantes infinitas
  (7, 49, 343…) y torres mixtas (8×5×7×11×13). Dentro de cada bloque se puede
  elegir libremente entre τ y su transpuesta: 2^328 soluciones de N=3080 sobre
  un mismo esqueleto.
- **Nivel 1 toroidal (dos espirales), probado como sii**: S es unión de clases
  del subgrupo `H = ⟨β/α, (α+1)/(β+1), (α−1)/(β−1)⟩`, con total
  `p²·Σ(2^k − 2)`. Regla exacta de la clase 3 tipo HAZ 2+1, y regla del doble
  reclamo para haces n+m (exacta con ≤6 clases).
- **Grupo de simetrías de las toroidales**: orden `8(p−1)p²`. Toda simetría
  propia contiene una de orden primo de dos tipos (escala o reflexión con eje
  de pendiente i o −1±√2).
- **Nunca se pueden mover 3 ni 5 reinas** entre toroidales (probado para todo
  primo p ≥ 7; el mínimo es 4).
- **Tres familias por N mod 6** cubren todo N de 4 a 600: paso constante
  (mcd(N,6)=1), paso 2 con un defecto (N par), pasos {N−3, N−2} (N≡3 mod 6).
  Son las construcciones clásicas redescubiertas desde los datos.
- **Familias con conteo exacto**: N≡2,4 mod 6 con salto `j=5+6m` da
  `N−7−18m` soluciones (verificado hasta N=1000); ley G para N par; tercios de
  paso 3; principio de bloques (`2g−1` tramos, mínimo 1, 3 o 5 según N mod 6).
- **Rachas de traslación**: regla exacta de qué desplazamientos de columnas
  siguen siendo solución. `W=2` pares de costura solo si N≡2,3,4 mod 6; por eso
  N=14, 15, 16 encadenan 11 giros. Con la esquina adecuada (según N mod 12) se
  alcanza N−3 en N=14, 20, 26, 32.

**Testigos y conteo**

- **Testigo de onda**: con la diagonal en la frecuencia,
  `conflictos = (⟨|W|²⟩ − N)/2`, exacto y entero. No hace falta clasificador.
- **Identidad de Vandermonde**: σ es solución sii `V(σ−id)·V(σ+id) ≠ 0`.
- **Forma epicíclica**: la curva de un tablero es invariante bajo el grupo afín
  `c_r → c_r + a(r+1) + b`, así que ningún rasgo de forma puede separar
  soluciones (techo de precisión 0,13 en N=8).
- **Polinomios exactos** `P_k(N)` para colocaciones parciales de k filas
  (k=2…7), con segundo coeficiente `−3k(k−1)/2`.
- **Arreglo de hiperplanos**: `χ_N(q)` cuenta N reinas en N filas del toro q×q;
  toda cámara de solución es acotada (N=4…9, sin prueba).
- **Una onda de rejilla levita exactamente las toroidales**; el mínimo de ondas
  por solución está calculado de forma exacta (MILP) para N=4…10 y casi todo
  N=11.
- **K_N sobre el toro** sin cruces exactamente para las soluciones toroidales
  (N=5, 7; la lineal de N=7 es el dual del mapa de Heawood).

## 4. Lo medido: señales reales

Efectos que sobreviven al control simétrico.

| Línea | Señal | Límite |
|---|---|---|
| `newton_queens/` | Desplazamiento medio ≈ N/2; cuencas modulares 1,2× mayores | Cosecha 4,7 % (N=8) → 0,6 % (N=11) |
| `wave_queens/` | N=7 modular: 71 % del tiempo plano, 14/28 soluciones | Estándar: techo de 1–2 soluciones por temperatura |
| `gcm_queens/` | Tasa de solución ~100× el azar junto a la transición | Residencia de 1 paso, sin itinerancia |
| `fase_queens/` | Aceleración común: cadenas de 10 vs 5 (N=8). N=16: dos grupos cerrados de 34 recorridos por una sola ley | La ley no reaparece en N=12 ni N=24 |
| `fase_queens/` (bucles) | N=9: ~775× el control; N=10: ~160× | El mecanismo es casi-toroidalidad (traslaciones) |
| `levitation_queens/` | Ondas mínimas ≈ nº de cortes de una toroidal (corr. 0,83) | Las reglas cubren 2–4 % de Q(N) en N=8–11 |
| `levitation_queens/` (fusión) | N=13: 99,5 % alcanzable en 7 niveles | Puertas al azar dan lo mismo; el nivel no ordena por sencillez |
| `poly_queens/` | El nivel de triples captura ~89 % de la constante de Simkin | La serie de cúmulos diverge (`R ≈ 4,9/N`): es el último nivel útil |
| `zeta_queens/` | Peine `2π(N+1)/(N+2)` autosemejante por filas; `1−(1−2f)⁴` de cobertura | Exceso ×0,5 por fila, azar desde la fila ~N−3 |
| `levitation_queens/` (perfil) | Curva de niveles con N=10…14 predice Q(15…27) dentro de 0,90–1,17 | Estima, no cuenta; sesgo ~3 % por N |
| `camino_queens/` | Lineales en bloque contiguo `[2N, (N−1)N)`; toroidales con centro como camino de monomios | Asimétricas de p=19 idénticas al azar en todos los niveles |
| `matrix/` | Grafo solo de swaps: 698 aristas vs 2,27 M en el nulo (N=16), sin explicar | La componente gigante es efecto de grados, no estructura |

## 5. Negativos que no hay que repetir

- **Mapas caóticos** (logístico, tienda, Hénon, gato, estándar): entre 1× y 4×
  la densidad una vez corregido el tamaño del repertorio. El caos da cobertura,
  no puntería.
- **Dinámica conservativa**: pasa más deprisa por donde V=0, así que cruza las
  soluciones sin detenerse. Las velocidades de equilibrio son independientes
  del acoplo: "quién va rápido" no puede ver las restricciones.
- **Potenciales de corto alcance**: mesetas planas, nada que seguir.
- **Las dos diagonales no son independientes**: `p_ambas/(p⁺·p⁻)` cae a 0,3
  desde N=10. "Resolver cada mitad e intersecar" no factoriza.
- **Sumar más niveles de cúmulos**: diverge.
- **Rasgos de forma tipo Fourier**: son invariantes afines, no separan.
- **Medias numéricas** (fracción continua, Khinchin, huecos): idénticas a las
  de permutaciones. La señal solo está en la cola, y la cola es aritmética
  (toda progresión mod N la tiene, sea solución o no).
- **Ondas más ricas** (≥1 armónico): ajustan igual de bien cualquier conjunto;
  la ventaja de las soluciones desaparece.
- **Agrupar por pendiente o por cortes** no predice qué pares se fusionan.
- **Sumar o combinar linealmente curvas** rompe la condición; lo único que
  funciona es elegir por fila qué curva manda.
- **Grafo de soluciones como poda** del backtracking: da información positiva y
  podar necesita negativa.
- **Receta fija de intercambios, islas copia, moldes XOR**: sin familias
  grandes; lo que se repite es giro + intercambio conmutando.

## 6. Lecciones de método

1. **Comparar siempre contra el control simétrico** (permutaciones con la misma
   órbita D4), no contra permutaciones al azar. El control simple infló varios
   efectos.
2. **Corregir por tamaño del repertorio**: una familia concentrada parece
   enriquecida sin estar alineada.
3. **Los N pequeños engañan**: el "binario" de Prony en N=7 y la pendiente
   `2^d` eran artefactos; la escalera se rellena en N=13.
4. **Mirar colas, no medias.**
5. **Contar grados de libertad del modelo**: la homogeneidad de un grupo la fija
   la riqueza de la onda, no las soluciones.
6. **Antes de proponer un invariante, comprobar bajo qué grupo es invariante.**
7. **Excluir toroidales y traslaciones triviales** antes de celebrar una cadena.
8. **Cuidado con los desempates** en búsquedas voraces: uno sesgado fabricó un
   falso "12×".

## 7. Mapa de carpetas

| Carpeta | Idea | Resultado en una línea |
|---|---|---|
| `spiral_queens/` | Espirales, Prony, toro de Clifford | Fórmula cerrada; la misma dicotomía en cinco lenguajes |
| `newton_queens/` | Partículas con F=ma y fricción | Funciona con heterogeneidad; ley N/2; caótico |
| `wave_queens/` | Osciladores con termostato | Intermitencia solo en modular; familia Sturmiana |
| `diag_waves/` | Diagonal en la frecuencia | Testigo exacto; mitades no independientes |
| `fractal_queens/` | Fractal de las reinas | Teorema `σ ⊗ τ`; árbol radial con pico en 0,7N |
| `fraction_queens/` | Soluciones como números en base N | Medias genéricas, cola = familia mod N |
| `poly_queens/` | Q(N) como función de partición | Triples 89 %, serie divergente |
| `arrangement_queens/` | Cámaras de hiperplanos | Cámaras de solución siempre acotadas |
| `levitation_queens/` | Ondas mínimas, reglas, fusión, toroidales por clase | La carpeta más grande; reglas exactas y censos |
| `torus_flow_queens/` | Líquido sobre el toro | Encadena con memoria; teorema de orden cíclico |
| `camino_queens/` | Orden de todos los tableros | Bloque contiguo para lineales; resto sin patrón |
| `zeta_queens/` | Conteo tipo π(x) | Peine diagonal autosemejante |
| `matrix/` | Grafo de Cayley de soluciones | Gigante trivial; anomalía de swaps |
| `swap_queens/` | Recetas de intercambio | Reflejos de clases de μ4/μ6; normales casi azar |
| `shape_atlas/` | Atlas de formas epicíclicas | Teorema de la órbita afín |
| `fase_queens/` | Una onda por reina, cadenas y bucles | Grupos de N=16; familias de traslación |
| `pendulum_queens/`, `landscape_queens/` | Péndulos, núcleo aprendido | 72/92 en N=8; sin fórmula natural |
| `gcm_queens/` | Mapas acoplados de Kaneko | Transición nítida, sin itinerancia |

## 8. Lo que queda abierto

Por si algún día se retoma. Cada una tiene una respuesta que no se conoce de
antemano.

- Probar que toda cámara de solución es acotada (criterio: grafo de pares
  cercanos conexo).
- Explicar la anomalía del grafo de swaps (698 vs 2,27 M).
- Esquina de las rachas según N mod 12: predicción N=38 → esquina A,
  N=44 → esquina B.
- Fórmula de recuento del haz 2+2 y por qué el solape empieza en 7 clases.
- Conjetura: el orden de Prony p=2 nunca aparece.
- Demostrar `R ≈ 4,9/N` para los ceros de `C_N(z)`.
- Itinerancia heteroclínica (variable lenta de fatiga) en lugar de caos de
  amplitud fija.
- Las nueve toroidales sin descripción de p=17 y p=19.

## 9. Regla para ideas nuevas

Antes de programar una idea, escribir en una línea **qué resultado sería una
sorpresa**. Si la respuesta esperada es "mod N o azar", se apunta aquí abajo y
no se ejecuta.

Ideas apuntadas:

- (vacío)

## 10. Qué hay en git

Están versionados el código, los README, las figuras y los datos pequeños.
Quedan fuera (ver `.gitignore`) el entorno `venv/`, los ejecutables compilados
y los datos pesados regenerables de `fase_queens/datos/`,
`fase_queens/barrido_N/datos/`, `levitation_queens/data/`,
`wave_queens/results/`, `shape_atlas/out/` y los volcados grandes de
`zeta_queens/`. Siguen en el disco local; cada uno se regenera con el programa
de su carpeta.
