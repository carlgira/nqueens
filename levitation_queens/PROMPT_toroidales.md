# Prompt para una conversación nueva: patrones en las toroidales de clase 3 en adelante

Trabajo en el repo `/Users/carlgira/files/nqueens/nqueens`, subproyecto `levitation_queens/`.
Python está en `venv/` (desde `levitation_queens/` se ejecuta con `../venv/bin/python`); hay
también programas en C ya compilados. Respóndeme siempre en español, claro y sin jerga; cuando
uses un término técnico, explícalo en una frase. Antes de afirmar un patrón, compáralo con un
modelo de azar y dime qué está comprobado y qué es solo observación.

## Objetivo

Encontrar reglas (idealmente fórmulas) para las soluciones **toroidales** de N-reinas con N
primo que necesitan **3 o más rectas**. Las de 1 y 2 rectas ya tienen fórmula; quiero saber si
las siguientes clases siguen algún patrón, cómo se relacionan unas clases con otras y cómo
cambian de un primo a otro.

## Definiciones

- **Solución**: lista T, T(fila) = columna de la reina, filas y columnas 0..p−1.
- **Toroidal**: sigue siendo válida con el tablero enrollado: T(r), T(r)+r y T(r)−r son
  permutaciones módulo p. (En la literatura, con una reina en el origen: *strong complete
  mapping* de Z_p.) Las toroidales son invariantes por traslación, así que todos los recuentos
  son múltiplos de p².
- **Recta / espiral**: T(r) = a·r + c (mod p), pendiente a en 2..p−2. Cada recta es una
  solución toroidal "lineal". En `super_queens.py` son espirales: (velocidad d, sentido s,
  giro i) ↔ T(r) = s·2d·(r+1) − i.
- **Clase** de una solución: número mínimo de rectas que cubren sus p reinas.

## Lo que ya está comprobado

Toroidales por clase (recorrido exhaustivo, los totales coinciden con OEIS A007705):

| p | total | clase 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 13 | 4.524 | 130 | 2.366 | 2.028 | – | – | – |
| 17 | 140.692 | 238 | 7.514 | 27.744 | 35.836 | 46.240 | 23.120 |
| 19 | 820.496 | 304 | 14.440 | 0 | 51.984 | 207.936 | 545.832 |

Divididos entre p² (soluciones por órbita de traslación): p=17 → 26, 96, 124, 160, 80;
p=19 → 40, 0, 144, 576, 1512; p=13 → 14, 12.

- **Clase 1**: p·(p−3).
- **Clase 2 toroidal, fórmula exacta** (comprobada hasta p=31): pendientes α, β; sea H el
  subgrupo de Z_p* generado por β/α, (α+1)/(β+1) y (α−1)/(β−1). La pareja sirve ⇔ H ≠ Z_p*,
  para cualquier desplazamiento; da 2^k − 2 soluciones por pareja, k = (p−1)/|H|, y las filas
  que toman la segunda recta son uniones de clases de H medidas desde la fila donde se cruzan.
  Recuentos: 0, 2.366, 7.514, 14.440, 16.928, 166.518, 230.640 para p = 11…31.
  En la literatura son los *quadratic / cyclotomic orthomorphisms*.
- **Clase 3 toroidal**: en p=13 y 17 todas tienen pendientes **(a, a, −a) con a² ≡ −1 (mod p)**
  (dos paralelas y una opuesta). Por eso p=19 (≡ 3 mod 4) no tiene ninguna. Combinando esas
  tres rectas salen 2.028 y 27.744; predice 4.167.996 (p=29), 58.845.096 (37), 199.232.120 (41),
  sin comprobar. **Falta**: la regla de qué filas toma cada recta (hoy se prueba por búsqueda).
- **Las rectas de clase ≥ 3 no pasan por un punto común**: vistas desde cualquier reina hacen
  falta 5 o más pendientes ("clase radial", `radial.py`). Por eso la fórmula de la clase 2 no
  se extiende tal cual.
- Familias "ciclotómicas" (una pendiente por clase de filas, todas las rectas por un punto):
  cubren el 100 % en p=13 con 6 pendientes, pero solo 34 % en p=17 (8 pendientes) y 24 % en
  p=19 (9 pendientes). Fear y Wanless (arXiv 2101.00859) prueban que siempre hay no ciclotómicas.
- No hay fórmula conocida para el total; crece como (N/e³)^N (Luria; Bowtell–Keevash,
  arXiv 2109.08083). Totales conocidos: A071607 × N.

## Cosas ya descartadas (no repetir)

- Fusionar soluciones por "niveles" con plantillas: funciona pero es lento y no ordena por
  sencillez.
- Trasladar una pareja concreta de espirales de un primo a otro (mismos giros o misma casilla
  de cruce): acierta igual que el azar.
- "Las no toroidales de clase 2 solo aparecen con β/α de orden ≤ 4": falso (falla en p=29, 31).
- Pocas rectas por sí solo no es estructura: una permutación al azar se cubre con solo una
  recta más que una solución.

## Herramientas que ya existen (en `levitation_queens/`)

- `minrectas.c` / `./minrectas p hilos`: mínimo de rectas de cada solución. Con `TORO=1` recorre
  solo toroidales; con `DUMPMAX=k` vuelca las de clase ≤ k a `data/clases_toro_p{p}.txt`
  (formato: `clase c0 c1 … c_{p−1}`). Ya volcadas completas para p = 13, 17, 19.
  `data/minrectas_toro_p{p}.txt` trae el histograma y, para clase ≤ 3, las pendientes de una
  cobertura mínima. **No guarda aún las pendientes de las clases 4–6**: habría que ampliarlo.
- `formula_espirales.py` (fórmula clase 2), `espirales.py` (mezcla de dos espirales por ciclos),
  `clase3_toro.py` (clase 3 por combinación), `radial.py`, `tabla_clases.py`,
  `clases_primos.py` (familias ciclotómicas con condición exacta), `lineas2.py`.
- `README.md` tiene el registro cronológico de todo; las últimas secciones son las relevantes.

## Qué me gustaría que intentaras, por orden

1. **Clase 3**: encontrar la regla de qué filas van en cada una de las tres rectas (a, a, −a).
   Mirar las 12 soluciones por órbita de p=13 y las 96 de p=17: posición de las tres rectas
   (el triángulo que forman, la distancia entre las dos paralelas) y el conjunto de filas de
   cada una. ¿Son clases de algún subgrupo desplazadas, progresiones, residuos cuadráticos
   respecto a dos centros distintos?
2. **Clases 4, 5 y 6** en p=17 y 19: sacar los patrones de pendientes de las coberturas mínimas
   (todas las coberturas, no solo una), como se hizo con la clase 3. ¿Hay pocas combinaciones,
   tipo (a, a, −a, −a) o (a, −a, 1/a, −1/a)? ¿Qué condición aritmética cumplen (a² ≡ −1,
   a³ ≡ ±1, relación con divisores de p−1)?
3. **Relación entre clases**: ¿una toroidal de clase m se obtiene de una de clase m−1 cambiando
   las filas de una recta? Comprobarlo solo entre toroidales y compararlo con el azar.
4. **Relación entre primos**: con las reglas que salgan, predecir p=23 (128.850.048 toroidales,
   se puede recorrer con `TORO=1 ./minrectas 23`) y comprobar.
5. Buscar si los números por órbita (26, 96, 124, 160, 80 en p=17; 40, 0, 144, 576, 1512 en
   p=19) tienen explicación: divisores de p−1, tamaños de grupos de simetría, etc.

Ve despacio, empezando por p=13 y p=17, y guarda cada resultado en el README y en la memoria.
