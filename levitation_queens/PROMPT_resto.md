# Prompt para una conversación nueva: las toroidales sin estructura — contarlas y cambiar de vista

Trabajo en el repo `/Users/carlgira/files/nqueens/nqueens`, subproyecto `levitation_queens/`.
Python está en `venv/` (desde `levitation_queens/` se ejecuta con `../venv/bin/python`); hay
programas en C ya compilados. Respóndeme siempre en español, claro y sin jerga; cuando uses un
término técnico, explícalo en una frase. Antes de afirmar un patrón, compáralo con un modelo de
azar y dime qué está probado, qué está comprobado exhaustivamente y qué es solo observación.

## Objetivo

Las toroidales de N-reinas (N primo) se dividen en un grupo "ordenado" (espirales, haces, mezclas,
simétricas), que ya sabemos describir, y un **resto sin simetría ni patrón** que crece hasta ser
casi todo. Esta conversación es sobre ese resto, en dos pasos:

1. **Contarlas**: cuántas son, cómo crece el número, y cuánto se parecen a "una colocación al azar
   que casualmente cumple las condiciones".
2. **Cambiar la perspectiva**: probar otras vistas, volúmenes, formas o geometrías (no el tablero
   plano ni las espirales) por si en otra representación se ve algo que aquí no.

## Punto de partida

Lee primero la última sección del `README.md` de `levitation_queens/`: "Niveles 0 y 1 definidos,
haces n+m, y el mapa completo de p=17 y p=19". Datos clave:

- Totales (OEIS A071607 × N, conocidos hasta N=31), en soluciones salvo traslación:
  36, 500, 2.288, 243.592, 720.472.148, 13.948.957.032 para p = 13, 17, 19, 23, 29, 31.
- Fracción con estructura: 100 % (13), ~62–74 % (17), 50 % (19), ~3 % (23), <0,01 % (29).
- Simetrías del toro: p² traslaciones × 8(p−1) (escalas y 8 giros). Una solución sin simetría
  propia tiene 8(p−1) copias salvo traslación.
- **Las nueve sin descripción**: p=17 índice 20; p=19 índices 13–18, 21, 22 (orden por (clase, tam)
  de `data/orbitas_p{p}.json`). Dibujadas en `fig_nueve.png` (`nueve.py`). Todas: sin simetría,
  5–6 rectas mínimas, no son mezcla de dos toroidales de ningún par, sí de tres piezas sencillas
  (indicio, azar 5–12 %).
- Volcados completos: `data/clases_toro_p{13,17,19}.txt`; p=23 parcial en `data/clases_toro0_p23.txt`.
- La literatura dice que no hay fórmula para el total y que crece como (N/e³)^N (Luria;
  Bowtell–Keevash, arXiv 2109.08083).

## Ideas para el paso 1 (contar)

- Separar por número de simetrías propias en p = 13, 17, 19, 23: ¿cuántas distintas hay sin
  ninguna simetría y cómo crece esa cuenta? ¿Se puede predecir p=29 y 31 restando al total de OEIS
  lo que tiene simetría?
- Comparar el resto con el azar en medidas sencillas (saltos entre filas, distancias entre reinas,
  número de rectas mínimas): ¿es indistinguible de una permutación al azar condicionada?
- Contar las de centro de simetría y otras simétricas directamente (problema de medio tablero)
  para p = 23, 29, 31.

## Ideas para el paso 2 (otras vistas)

En el repo hay vistas ya construidas que se pueden reutilizar sobre las nueve y sobre muestras del
resto de p=23: `diag_waves/` (ondas, espectros, toro, esfera, cilindro), `spiral_queens/` (esfera,
fibración de Hopf), `arrangement_queens/` (cámaras de hiperplanos), `fraction_queens/`,
`poly_queens/`. En esta misma carpeta: `cilindro.py`, `espirales_log.py` (rejilla de potencias),
`dos_polos.py` (tabla de dos cruces). Posibles vistas nuevas: el toro en 3D con las tres
direcciones prohibidas, el espectro (transformada de Fourier de la solución), la solución como
camino en el grupo de simetrías, distancias a las soluciones ordenadas más cercanas.

## Cosas ya descartadas (no repetir)

- Describirlas con pocas espirales: con 5–6 también se cubre una colocación al azar.
- Mezcla de dos toroidales: ninguna lo es. Encadenar mezclas tampoco llega.
- Grado del polinomio: casi siempre máximo, no distingue.
- Parecido visual en el tablero: el arranque común (0, 2, 4) es efecto del representante elegido.
  Parecido real entre las 8 de p=19: 9,4 reinas en común de 19 frente a 8,0 con el resto.

Guarda cada resultado en el README y en la memoria.
