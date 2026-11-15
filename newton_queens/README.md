# Newton Queens

**Las N-Reinas resueltas como un sistema físico continuo, caótico y disipativo.**

En vez de buscar en un grafo discreto (backtracking, SAT, CP), soltamos `N`
partículas en un plano 2D y las dejamos evolucionar bajo **F = m·a**. Las reglas
del ajedrez se codifican como campos de energía potencial continuos. El estado
de reposo del sistema —todas las fuerzas anuladas— *es* una solución.

```
m_i r_i'' = -∇U_grid(r_i)  -  Σ_{j≠i} k_j ∇U_rep(i,j)  -  γ r_i'
```

## Los tres campos

**1. El tablero — `U_grid` (cartón de huevos).** Un terreno de ondas de Fourier
que atrae cada partícula hacia las coordenadas enteras (las casillas válidas):

```
U_grid(x,y) = -A ( cos(2πx) + cos(2πy) )
```

Los mínimos están en cada par de enteros; una partícula que cae rueda hacia la
casilla más cercana.

**2. Las reinas — `U_rep` (repulsión anisotrópica).** No es Coulomb: una reina
no ataca en círculos. La repulsión tiene forma de estrella y solo se dispara
cuando dos reinas se alinean en fila, columna o diagonal. Pozos gaussianos sobre
las 4 proyecciones (`g(s)=exp(-s²/2σ²)`, `dx=xᵢ-xⱼ`, `dy=yᵢ-yⱼ`):

```
U_rep(i,j) = k_j [ g(dx) + g(dy) + g(dx-dy) + g(dx+dy) ]
```

`σ` es el ancho del "rayo láser" de la reina. Pequeño → solo se repelen si están
casi perfectamente alineadas.

**3. Disipación — fricción viscosa `-γ·v`.** Roba energía cinética. El sistema es
disipativo: el caos pierde energía hasta detenerse en un atractor.

## El hallazgo: la formulación homogénea es degenerada

Al implementar el sistema *tal cual*, **nunca** converge a una solución válida de
N=8 (0/480 en el barrido de parámetros). No es un problema de tuning: es
estructural.

> El gradiente de una gaussiana **se anula en su cima**. Dos reinas perfectamente
> alineadas (misma fila/columna) forman un punto de fuerza de separación *nula*:
> están en el peor sitio energético posible, pero ninguna fuerza las empuja a
> salir. La red densa de pozos las clava ahí. El sistema se congela con 2–4
> conflictos, siempre.

## La cura: romper la homogeneidad

En vez de meter temperatura (recocido), se rompe la simetría dando a cada reina
una **identidad propia** (idea de heterogeneidad):

- **Energía cinética inicial aleatoria** (`v0`): cada reina arranca con inercia.
  La que cae en un mínimo falso lleva impulso para subir la colina y escapar —
  como un péndulo con velocidad de sobra.
- **Poder individual `k_i`**: cada reina proyecta un campo de fuerza propio. Las
  poderosas empujan con violencia; las débiles apenas afectan. La fuerza sobre
  `i` va ponderada por el poder `k_j` de quien la empuja.
- **Masa acoplada al poder** (`m_i ~ k_i`): las poderosas son pesadas e
  inamovibles; las débiles, ligeras y salen despedidas. Emerge una *jerarquía* —
  los cuerpos poderosos reclaman sus dimensiones y barren a los débiles hacia las
  zonas vacías. Es caos, pero un caos jerárquico.

Con **fricción baja** las reinas vuelan balísticamente por el cartón de huevos,
rebotando y muestreando muchas configuraciones; atrapamos la solución en el
instante en que la configuración redondeada es válida y ahí la congelamos.

## Basin hopping: 10.000 universos

Un solo universo resuelve pocas veces (~5–7% para N=8). Pero la idea gana con la
repetición: lanzamos miles de universos con posición y energía aleatorias,
**cosechamos** los que cristalizan en una solución real y **desechamos** los que
se atascan. Toda la física está vectorizada sobre el eje de universos, así que
los 10.000 se integran a la vez en numpy.

```bash
python experiment.py 8 10000     # N=8, 10.000 universos → estadísticas + figuras
```

Produce: tasa de éxito, distribución de pasos hasta cristalizar, número de
soluciones **distintas** descubiertas (y cuántas de las 12 fundamentales de N=8),
y dónde se paralizan los que fallan. Guarda `experiment_N8.png` y
`experiment_N8_samples.png`.

### Cosecha de N=8 (10.000 universos)

- **474 soluciones válidas (4.74%)**; los 9.526 restantes se atascan con 2–4
  conflictos (la firma de la degeneración).
- **91 de las 92** soluciones exactas descubiertas, y **12 de las 12** clases
  fundamentales (100%). El azar dinámico barre casi todo el espacio de soluciones.
- Mediana de ~5.300 pasos de integración hasta cristalizar.

## Análisis de trayectorias (`analyze.py`)

Buscando estructura en el caos. Hallazgos sobre N=8:

**Equivariancia D₄ vs caos.** El potencial es invariante bajo las 8 simetrías del
cuadrado y la fricción es isótropa ⇒ la dinámica es **equivariante**: las fuerzas
cumplen `F(g·x) = g·F(x)` a precisión de máquina (~5e-14). En teoría, rotar la
condición inicial (posiciones *y* velocidades) lleva a la solución rotada. **Pero
el sistema es caótico**: la identidad y la transposición (`reflex \`) son exactas
en binario (solo permutan/niegan ejes) y se mantienen a error 0; las demás
simetrías introducen redondeo de ~1e-16 que el caos **amplifica exponencialmente**
(exponente de Lyapunov **λ ≈ 3.3**, error que se duplica cada ~17 pasos) hasta
cruzar una frontera de cuenca y caer en *otra* solución válida. A corto plazo las
8 imágenes forman un mandala D₄ perfecto; a largo plazo divergen.

**Poder vs movimiento.** La hipótesis de la heterogeneidad se confirma con fuerza:
`corr(poder k, longitud del camino) = −0.81`, `corr(masa, camino) = −0.87`. Las
reinas poderosas y pesadas reclaman su sitio y apenas se mueven (camino ~150); las
débiles vagan más del doble (~345). Curiosamente el poder predice *cuánto vagan*,
no *dónde acaban* (`corr(k, desplazamiento) ≈ 0`).

**Condición inicial vs final.** Prácticamente **decorreladas** (`corr ≈ 0.05`): el
caos borra la memoria de dónde empezó cada reina. Desplazamiento medio ~3.8
casillas (media tabla). Más conflictos/energía inicial → algo más de pasos
(`r ≈ +0.2`), pero la tasa de éxito apenas depende del punto de partida.

**Cuencas simétricas.** Las 12 clases fundamentales tienen cuencas de **tamaño muy
distinto** (de 40 a 5 cosechas), pero dentro de cada clase las ~8 orientaciones
salen con frecuencia parecida — justo lo que predice la equivariancia: soluciones
relacionadas por simetría tienen cuencas congruentes.

## Estudio de cuencas de atracción (`basin_study.py`)

Barriendo N=7 (primo, alto rendimiento: 14.6% y las 40 soluciones cubiertas):

**Tamaño de cuenca.** Cada solución es el destino de decenas de condiciones
iniciales distintas: de 18 a 74 (mediana 55), una desigualdad de **4.1×**.

**¿Posición o energía?** Fijando una y variando la otra: desde **una sola
posición inicial**, variando solo las energías, se alcanzan **39 de 40**
soluciones; desde **una sola energía**, variando posiciones, **38 de 40**. El caos
scramblea tanto el mapa inicial→final que cualquiera de las dos fuentes de azar
basta para recorrer casi todo el espacio.

**Ley del desplazamiento (universal).** La reina media acaba a `0.97–1.01 × (N/2)`
casillas de donde empezó, *el mismo factor* para N=7 (3.40), N=8 (3.86) y N=11
(5.58). El caos borra la memoria del origen a media tabla, sin importar el tamaño.

**Estructura de los primos.** Para N coprimo con 6 existen las soluciones
**modulares/toroidales** (`perm[i]=a·i+b mod N`). En N=7, 28 de las 40 lo son, y el
sistema físico **las prefiere**: cuenca media modular 53.7 vs 44.8 (mediana 58 vs
37, **1.2×**). Las soluciones más estructuradas atraen más trayectorias.

En N=11 (donde solo 88 de 2680 son modulares, 3.3%) el rendimiento cae al ~0.66%:
con 118 cosechas las modulares salen al 2.5% (3/118), **indistinguible de la tasa
base** (enriquecimiento 0.77×; esperado ~3.9, observado 3). La preferencia por la
estructura es clara en el régimen denso de N=7, pero en N=11 el método está
subpotenciado — detectar un 1.2× sobre una base del 3.3% exigiría miles de cosechas
(horas de cómputo). Queda como pregunta abierta, no refutada.

## Archivos

| archivo | qué es |
|---|---|
| `newton_queens.py` | motor de simulación (un universo): fuerzas, RK4, heterogeneidad |
| `experiment.py` | basin hopping vectorizado de 10.000 universos + estadísticas |
| `analyze.py` | análisis de trayectorias: equivariancia/Lyapunov y patrones |
| `basin_study.py` | cuencas de atracción: tamaños, anclas, ley N/2, estructura primos |
| `animate.py` | animación en vivo con matplotlib (basin hopping visual) |
| `index.html` | simulación interactiva en el navegador (tocar constantes en caliente) |
| `solutions_N8.txt` | soluciones distintas cosechadas por el experimento |

## Uso

```bash
python newton_queens.py 8        # resuelve un tablero por reinicios, lo imprime
python animate.py 8              # ventana animada
python animate.py 8 --gif out.gif
python experiment.py 8 10000     # el experimento de basin hopping
```

Abre `index.html` en el navegador para la versión interactiva: mueve `A`, `B`,
`σ`, `γ` y `T₀` en caliente y observa cómo cambia el atractor.
```

Requisitos: `numpy`, `matplotlib`.
