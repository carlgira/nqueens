# shape_atlas — la forma de un tablero dibujado con epiciclos

Cada tablero con una reina por fila (columna `c_r` en la fila `r`, `r = 0..N−1`) se
convierte en una cadena de N vectores giratorios, como en el vídeo de 3Blue1Brown:

```
E0 (fila → frecuencia, columna → fase):   z(t) = Σ_r exp(i·((r+1)·t + 2π·c_r/N)),   t ∈ [0, 2π)
```

La idea era mirar de abajo arriba las formas de soluciones y tableros malos y buscar
un rasgo de la forma que las separe. **Respuesta corta: en E0 no hay ningún rasgo de la
forma que distinga las soluciones, y se puede demostrar que no puede haberlo.** La
separación estadística es débil (AUC 0,61 a 0,85). Solo hay criterios exactos si se mira
la curva *con su parámetro* (puntos marcados `z(2πk/N)`), y esos solo hablan de
diagonales **módulo N** (el problema toroidal). Cuando las diagonales se meten **en la
frecuencia** (E2), el área encerrada sí es un testigo exacto, pero ese es el testigo de
`diag_waves/` visto como área.

Galería: [`gallery/atlas.html`](gallery/atlas.html), un solo archivo que se abre sin servidor.
Tiene las 92 soluciones de N=8 por clase fundamental junto a 3×92 controles, los «gemelos»
(misma forma con distinto estatus), las flores lineales de N=7 y 11, toroidales no lineales
de N=13, histogramas de rasgos y la tabla de rasgos de las 92 soluciones. Se puede
cambiar entre las codificaciones E0/E1/E2.

---

## Cómo reproducir

```
cc -O2 -o atlas atlas.c -lm
./atlas facts      > out/facts.txt          # verificaciones (a), (b), (c), Parseval   (~8 s)
./atlas stats 8    > out/stats_N8.txt       # rasgos de las 40 320 permutaciones + 5000 al azar (~2 min)
./atlas stats 9    > out/stats_N9.txt       # 352 sol + 40 000 perms al azar + casi-soluciones
./atlas stats 10   > out/stats_N10.txt
./atlas joint N    >> out/joint.txt         # simetría especular y forma conjunta E0+E1
./atlas gallery    > gallery/data.js        # datos de la galería
gallery/build.sh                            # → gallery/atlas.html (autocontenido)
```

`stats` también escribe `out/feat_N*.csv`, con todos los rasgos de todas las muestras (aquí van comprimidos en `.gz`).
Todo el cálculo está en C. Los únicos usos de Python fueron editar texto y empaquetar `texts.js`.

**Rasgos calculados** sobre 4099 muestras por curva (número primo, así ninguna muestra cae
en un `2πk/N`). Los mínimos de |z| y de |z'| se refinan por sección áurea.
Longitud de arco `L`, área algebraica, |z| mín./máx., `⟨|z|²⟩`, `⟨|z|⁴⟩`, anisotropía y
centroide del trazo, curvatura absoluta total, velocidad mínima, pasos exactos por el
origen (`nzero`) y cuántos de ellos caen en `t=2πk/N` (`nzlat`), número de giro, número de
rotación de la tangente, autointersecciones, lóbulos, inflexiones y cúspides. Además,
`zt1 = |z(2π/N)|` como ejemplo de *punto marcado*, que no es un rasgo de la forma.

**Codificaciones:**

| | frecuencia | fase |
|---|---|---|
| E0 | fila `r+1` | columna `2πc_r/N` |
| E1 | columna `c_r+1` | fila `2πr/N` |
| E2 | tres bandas: `r+c+1` (antidiagonal), `c−r+3N−1` (diagonal), `c+4N−1` (columna) | 0 |

---

## A. Lo EXACTO

### A1. La forma de E0 no ve una cizalla del tablero (teorema, y la causa de todo lo demás)

```
c_r → c_r + a·(r+1) + b  (mod N)      ⟹      z(t) → e^{2πib/N} · z(t + 2πa/N)
c_r → −c_r                            ⟹      z(t) → conj(z(−t))   (espejo)
```

Cizallar el tablero solo **adelanta el reloj** (cambia dónde está t=0), y girar las
columnas cíclicamente solo **gira** la figura. Así que todo rasgo de la forma (cualquiera
que no dependa de dónde empieza t ni de la orientación) toma el mismo valor en los N²
tableros de la órbita `G = {c + a(r+1) + b}`. También se cumple lo recíproco. Si dos tableros
dibujan la misma imagen salvo giro, la reparametrización tiene que ser una traslación
de t, y como las fases viven en 2πℤ/N con frecuencias consecutivas 1..N, esa traslación es
un múltiplo de 2π/N. **La forma de E0 es exactamente la órbita bajo el grupo afín de
cizallas.** Para comprobarlo numéricamente, una solución y su cizalla+giro dan los mismos
rasgos salvo discretización (ver C).

Lo que eso significa para las N-reinas estándar (permutaciones con solución dentro de la órbita):

| N | | órbitas con solución | permutaciones en ellas | soluciones | techo de precisión de *cualquier* clasificador por forma |
|---|---|---|---|---|---|
| 8 | curva idéntica, misma orientación (solo `a`) | 92 | 130 | 92 | 0,708 |
| 8 | forma salvo giro (`a`, `b`) | 62 | 704 | 92 | **0,131** |
| 9 | solo `a` | 352 | 768 | 352 | 0,458 |
| 9 | `a`, `b` | 226 | 3906 | 352 | **0,090** |
| 10 | solo `a` | 724 | 748 | 724 | 0,968 |
| 10 | `a`, `b` | 548 | 5600 | 724 | **0,129** |

Añadir el espejo no cambia nada: el espejo de una solución es otra solución. Con N=8, 38 de
las 92 soluciones tienen un tablero malo que dibuja **exactamente la misma curva, con la misma
orientación**. Salvo giro, las 62 formas de las 92 soluciones también las dibujan 612
permutaciones que no son solución. La sección 2 de la galería enseña estos gemelos.
**No hay criterio exacto por forma en E0, y tampoco en E1** (E1 es E0 aplicado al tablero
transpuesto).

### A2. Hecho (a): `z(0)=0 ⇔ una reina por columna` solo vale para N primo

`z(0) = Σ_j m_j ω^j`, donde `m_j` es el número de reinas en la columna j. Se enumeraron todos
los vectores `(m_0..m_{N−1})` con `Σm = N` (ver `out/facts.txt`):

| N | ¿primo? | tableros con z(0)=0 | N! | ¿«si y solo si»? |
|---|---|---|---|---|
| 5, 7, 11 | sí | = N! | | **sí** |
| 4 | no | 36 | 24 | no |
| 6 | no | 2040 | 720 | no |
| **8** | no | **190 120** | 40 320 | **no**: 149 800 tableros con columnas repetidas cumplen z(0)=0 |
| 9 | no | 640 080 | 362 880 | no |
| 10 | no | 32 458 860 | 3 628 800 | no |
| 12 | no | 17 365 421 304 | 479 001 600 | no |

Para N=4..9 se comprobó además por fuerza bruta sobre los N^N tableros (coincide). Con N=8,
la condición es que las columnas opuestas `j` y `j+4` tengan la misma ocupación (35 vectores
de multiplicidad; las sumas mínimas que se anulan entre raíces 8-ésimas son los pares
antipodales). La dirección ⇒ vale siempre.

### A3. Hecho (b): `toroidal ⇔ z(0) = z(±2π/N) = 0` solo vale para N primo

`z(2πk/N) = Σ_r ω^{c_r + k(r+1)}`, así que los tres ceros dicen que las columnas, `c+r` y
`c−r` están equidistribuidas mod N en el sentido de las sumas de raíces que se anulan.

| N | tableros con los tres ceros | toroidales | discrepancias |
|---|---|---|---|
| 5 | 10 | 10 | 0 |
| 7 | 28 | 28 | 0 (fuerza bruta sobre 7⁷) |
| 11 | 88 | 88 | 0 (sobre las 11! permutaciones; z(0)=0 ⇒ permutación está probado en A2) |
| 4 / 6 / 8 / 9 / 10 | 4 / 18 / **312** / 54 / 300 | 0 | todas |

Para N=8 no existe ninguna solución toroidal, pero 312 tableros cumplen las tres anulaciones.
⇒ vale siempre y ⇐ solo para N primo.

### A4. Hecho (c): la solución lineal da un núcleo de Dirichlet, no una flor simétrica

```
c_r = a·r mod N   ⟹   z_a(t) = e^{it}(1 − e^{iNt}) / (1 − e^{i(t+2πa/N)}),   |z_a| = |sin(Nt/2) / sin((t+2πa/N)/2)|
```

Es cierto que tiene **N−1 pasos por el origen y N−1 lóbulos** (verificado con N=7, 8 y 11 y
todas las a), pero no es una flor regular: un lóbulo grande (|z|=N) y N−2 lazos pequeños
cerca del origen. Además, por A1, **todas las a dan la misma curva girada**, y eso incluye
a=0 (todas las reinas en una columna) y a=±1 (una diagonal), que no son solución. La flor no
sabe si el tablero es solución. Con N=7 y 11 todas las soluciones toroidales son lineales
(28 = 4·7 y 88 = 8·11), así que todas dibujan la misma figura. Con N=13 aparecen toroidales
no lineales con otras formas (galería, sección 3).

### A5. Otros hechos exactos de E0

- **Área (Parseval)**: `área = π·Σ(r+1) = π·N(N+1)/2` para todo tablero con una reina por
  fila. Medido en 2000 tableros al azar, el error máximo es 1,7·10⁻¹². También
  `⟨|z|²⟩ = N` siempre.
- **Simetrías de la figura**: ninguna curva E0 tiene simetría de rotación. Las frecuencias
  1 y 2 están presentes y obligan a que el giro sea de 2π. La simetría especular ocurre
  ⇔ `−c ∈ G·c`, es decir, ⇔ el tablero es lineal cuando N es impar. Medido: N=7 da 28/40
  soluciones (las lineales), N=11 da 88/2680, y N=8, 9, 10 y 12 dan **0**.
- **Paso por el origen**: toda permutación pasa por el origen en t=0. Los tableros al azar
  casi nunca pasan.

### A6. E2 (diagonales en la frecuencia): el área sí es un testigo exacto

Con fase 0, `área = π·Σ_k k·m_k²`, donde `m_k` es la ocupación de la frecuencia k. Entre las
permutaciones, `Σ_k k·m_k` es constante, así que

```
área − área_sol = π · Σ_k k·m_k(m_k−1)  ≥ 0,   con igualdad ⇔ solución
```

| N | área de toda solución | exceso mínimo con un conflicto |
|---|---|---|
| 8 | π·524 (las 92 idénticas) | 6π |
| 9 | π·666 | 6π |
| 10 | π·825 | 6π |

Además, `⟨|z|²⟩ = 3N + 2·(pares en conflicto)` para cualquier tablero con una reina por fila.
Es el testigo de `diag_waves/` y vale incluso sin ser permutación. El área es la versión
geométrica de ese testigo: no depende de la parametrización. Un efecto secundario exacto
(N par, permutaciones): `z(π) = 2·Σ_r (−1)^{r+c_r+1}`, así que la curva E2 pasa por el
origen en t=π ⇔ hay tantas reinas en casillas blancas como en negras. Medido: las 92
soluciones de N=8 están equilibradas (balance 0) y las 724 de N=10 tienen balance ±2.

### A7. Forma conjunta E0 + E1

Si se exige a la vez la forma fila→frecuencia **y** la forma columna→frecuencia, la
ambigüedad de A1 desaparece casi por completo (`out/joint.txt`). Con N=8, 9, 10 y 12, ningún
tablero malo comparte las dos formas con una solución. Con N=7 y 11 solo quedan como gemelos
las soluciones lineales (sus gemelos son las diagonales `c = ±r + b`). Esto dice que el par
de formas casi determina el tablero, **no** que haya un rasgo simple. Para «leer» el estatus
haría falta compararlo con la lista de pares de formas de las soluciones. (Esto da por buena
la inyectividad imagen → órbita de A1, que se argumenta pero no se ha probado para curvas
no genéricas.)

---

## B. Lo ESTADÍSTICO

AUC de cada rasgo para soluciones contra permutaciones que no son solución. 0,5 significa
que el rasgo no separa nada; 0 o 1, que separa del todo. N=8 usa las 40 320 permutaciones y
N=9 y 10, todas las soluciones más 40 000 permutaciones al azar (`out/stats_N*.txt`).
Debajo, los rasgos más informativos de E0:

| rasgo E0 | N=8 | N=9 | N=10 | lectura |
|---|---|---|---|---|
| `nzero` (pasos por el origen) | 0,67 | **0,85** | 0,51 | las soluciones pasan más veces por el origen (2,9 frente a 1,1 en N=9) |
| `nzlat` (pasos en t=2πk/N) | 0,61 | 0,83 | 0,49 | casi todo `nzero` cae en la red |
| `zt1` = \|z(2π/N)\| (marcado) | 0,36 | 0,28 | 0,32 | la solución pone ≤2 reinas por clase `c+r mod N` |
| `L` (longitud) | 0,42 | 0,37 | 0,39 | soluciones algo más cortas |
| `m4` = ⟨\|z\|⁴⟩ | 0,53 | 0,65 | 0,67 | |
| resto (autointersecciones, lóbulos, curvatura, giro, cúspides…) | 0,44–0,55 | 0,50–0,60 | 0,46–0,57 | ruido |
| **LDA con 13 rasgos geométricos (validación cruzada 5-fold)** | **0,61** | **0,85** | **0,65** | |

- El rango de valores de las soluciones contiene al 85–100 % de las permutaciones malas
  en cualquier rasgo de E0. Ningún umbral separa.
- La ventaja de N=9 viene de la aritmética de 9 = 3². Las sumas de raíces 9-ésimas se
  anulan por ternas `{j, j+3, j+6}`, y las soluciones cumplen esa equidistribución mod 3
  más a menudo, en varias pendientes k. Es un efecto de `z(2πk/N)`, o sea de los pasos por
  el origen, y no de la silueta.
- **E1** tiene exactamente la misma distribución que E0 sobre las permutaciones, porque
  transponer conserva las soluciones. No aporta nada estadístico por sí sola (aunque sí
  junto a E0, A7).
- **E2**: `L` (AUC 0,004–0,006) y `⟨|z|⁴⟩` (0,001–0,006) separan casi del todo, pero no de
  forma exacta (el 2–3 % de las permutaciones malas cae dentro del rango de longitudes de
  las soluciones). La LDA llega a 0,98–0,99. Solo área y `⟨|z|²⟩` son exactos. Contra las
  casi-soluciones (un conflicto) la longitud baja a AUC ≈ 0,14–0,19.

---

## C. Avisos numéricos

- Las **autointersecciones** pueden cambiar en ±2 entre tableros de la misma órbita en
  ~2–5 % de los casos. Hay contactos casi tangentes entre ramas, y uno real: dos pasos por
  el origen con la misma tangente, que aparecen en soluciones de N=8 con ceros en t=0 y
  t=π. Los cruces en el origen se cuentan aparte, como C(nzero, 2).
- La **curvatura absoluta total** diverge cerca de cúspides (|z'|→0). Por eso la media de
  `perm` en N=10 sale 10⁷; el AUC no se ve afectado.
- `rmax` se mide por muestreo, con un error relativo de ~10⁻⁵.

---

## D. Conclusión

1. **No existe ningún rasgo de la forma de E0 (ni de E1) que distinga las soluciones del
   N-reinas estándar.** No es que no se haya encontrado: está prohibido. La forma es un
   invariante de la órbita afín `c + a(r+1) + b`, y en esas órbitas hay de 6,7 a 10 permutaciones
   malas por cada solución.
2. Lo que **sí** es exacto en E0 son **puntos marcados del reloj**, los valores `z(2πk/N)`.
   Esos valores solo saben de rectas **mod N**: columnas, y diagonales toroidales para N
   primo. Para el problema estándar, cuyas diagonales no son modulares, no dan ningún
   criterio exacto.
3. Estadísticamente, la señal está en el número de pasos por el origen y en la longitud.
   Es débil (AUC 0,6–0,85) y depende de la aritmética de N.
4. El criterio exacto aparece al cambiar de codificación y poner **las diagonales en la
   frecuencia** (E2). Ahí el área encerrada es constante para las soluciones y estrictamente
   mayor con cualquier conflicto. Es el mismo testigo cuadrático de `diag_waves/` y, como
   allí, *describe* sin *construir*.

**Pista para la búsqueda «de abajo arriba»**: el objeto natural no es la curva sino la curva
módulo traslación del reloj y rotación. Eso equivale al tablero módulo el grupo afín
`ℤ_N ⋉ ℤ_N`, el mismo grupo que organiza las soluciones toroidales/lineales en el resto del
repo. Una teoría de curvas que «genere» soluciones tendría que vivir en ese cociente (por
ejemplo, invariantes de curvas trigonométricas bajo traslación de parámetro) y además
recuperar la información no modular de las diagonales. E0 no la tiene. E2 sí.
