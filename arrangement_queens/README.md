# arrangement_queens — el espacio de todas las soluciones

Inspiración: el vídeo de 3Blue1Brown sobre el rectángulo inscrito, donde el
problema se pasa a un espacio de configuraciones (una banda de Möbius) y la
topología de ese espacio fuerza la solución.

Para las N reinas la topología **no puede** forzar la existencia (N=2 y N=3 no
tienen solución). Pero sí existe un espacio donde viven todas las soluciones a la
vez.

**Resumen en una frase**: con columnas reales `x_i`, cada amenaza es un hiperplano
de ℝᴺ. Las soluciones son los puntos enteros de `{0..N-1}ᴺ` que caen dentro de
alguna **cámara** del complemento. Contar cámaras es un problema limpio (un
polinomio), pero Q(N) no lo es.

---

## 1 · La disposición `A_N`

```
x_i - x_j = 0          (columna)
x_i - x_j = i - j      (diagonal)
x_i - x_j = j - i      (antidiagonal)        para todo i < j   ->   3·C(N,2) hiperplanos
```

Todos contienen la dirección `(1,…,1)`, así que se trabaja módulo traslación
(dimensión N−1). Contiene la disposición de trenzas `x_i = x_j`, así que dos
soluciones distintas siempre están en cámaras distintas.

## 2 · Polinomio característico · `charpoly.py`, `torus_count.c`

Método de cuerpos finitos: `χ_N(q) = #{x ∈ Z_qᴺ fuera de los hiperplanos}`, es
decir, **N reinas en N filas de un toro q×q**. La matriz de coeficientes es gráfica
(unimodular), así que la reducción es buena para primos q mayores que la máxima
suma de ganancias de un ciclo (~N²/2). Se interpola con N primos y **se verifica con
2 primos más**. Zaslavsky: cámaras = |χ(−1)| y acotadas = |χ(1)|.

| N | χ_N(q) | cámaras | acotadas | Q(N) |
|---|--------|--------:|---------:|-----:|
| 2 | `q^2 - 3q` | 4 | 2 | 0 |
| 3 | `q^3 - 9q^2 + 22q` | 32 | 14 | 0 |
| 4 | `q^4 - 18q^3 + 119q^2 - 286q` | 424 | 184 | 2 |
| 5 | `q^5 - 30q^4 + 367q^3 - 2170q^2 + 5232q` | 7,800 | 3,400 | 10 |
| 6 | `q^6 - 45q^5 + 873q^4 - 9159q^3 + 52082q^2 - 128494q` | 190,654 | 84,742 | 4 |
| 7 | `q^7 - 63q^6 + 1767q^5 - 28371q^4 + 276054q^3 - 1547306q^2 + 3907356q` | 5,760,918 | 2,609,438 | 40 |
| 8 | `q^8 - 84q^7 + 3210q^6 - 72678q^5 + 1057285q^4 - 9915384q^3 + 55627262q^2 - 144148022q` | 210,823,926 | 97,448,410 | 92 |

## 3 · La cámara de cada solución · `chambers.py`

Para N=4…9 (500 soluciones), con LP:

- **Todas las cámaras de soluciones están acotadas** (100 %), frente al 43–46 % del
  total de cámaras. El criterio es exacto y combinatorio: el grafo de pares
  «cercanos» `|x_i−x_j| < |i−j|` tiene que ser conexo. Las filas 0 y N−1 siempre son
  cercanas, porque `|x_0−x_{N−1}| = N−1` sería un ataque en diagonal.
- **Paredes que tocan la cámara**: de media 8, 10.4, 15, 16.6, 19.1, 21.9 para
  N=4…9. Crece como ~2.4·N, mientras que el total de paredes crece como 1.5·N².
- **Profundidad**: toda solución está a 1/√2 de su pared más cercana, lo mínimo
  para un punto entero, y casi siempre ese es también el inradio de la cámara.
  **Excepción: N=6**, donde `1 3 5 0 2 4` y `4 2 0 5 3 1` tienen inradio 0.884 y
  sus traspuestas no. La disposición distingue filas (índices) de columnas
  (valores), así que rompe la simetría de rotación del tablero.
- En N=7 las soluciones toroidales tocan más paredes (16.9) que las no
  toroidales (16.0).

## 4 · Visualización · `index.html` (se genera con `build.py` desde `template.html`)

- **Espacio completo** (N=3, 4): ℝᴺ módulo traslación. Las N! permutaciones
  forman un permutoedro. En N=3 son un hexágono con los 6 vértices sobre alguna
  de las 9 rectas, de ahí que no haya solución. En N=4 hay 18 planos, 424 cámaras
  y solo 2 vértices libres, cada uno dentro de su poliedro.
- **Corte 3D** (N=4…10): se liberan 3 filas y se fijan las demás. Se ve la celda de
  la solución y las otras soluciones que solo cambian esas 3 reinas.

## Abierto

- ¿Por qué las cámaras de soluciones son siempre acotadas? Falta una prueba de que
  el grafo de pares cercanos es conexo para toda solución.
- Complejificar: `ℂᴺ` menos los hiperplanos es conexo, y su π₁ serían las
  «trenzas de reinas». El álgebra de Orlik–Solomon da su cohomología.
