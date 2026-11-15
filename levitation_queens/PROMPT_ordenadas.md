# Prompt para una conversación nueva: completar las toroidales "ordenadas" y generarlas directamente

Trabajo en el repo `/Users/carlgira/files/nqueens/nqueens`, subproyecto `levitation_queens/`.
Python está en `venv/` (desde `levitation_queens/` se ejecuta con `../venv/bin/python`); hay
programas en C ya compilados. Respóndeme siempre en español, claro y sin jerga; cuando uses un
término técnico, explícalo en una frase. Antes de afirmar un patrón, compáralo con un modelo de
azar y dime qué está probado, qué está comprobado exhaustivamente y qué es solo observación.

## Objetivo

Entender del todo las toroidales de N-reinas (N primo) que tienen estructura, de forma que se
puedan **generar directamente, sin buscar**, y ver hasta dónde llegan las reglas generales
(válidas para cualquier primo). No se trata de explicar todas las toroidales: el resto sin
estructura va en otra conversación.

## Punto de partida

Lee primero la última sección del `README.md`: "Niveles 0 y 1 definidos, haces n+m, y el mapa
completo de p=17 y p=19", y la anterior ("Toroidales de clase >= 3…"). Resumen:

- **Una espiral** (recta T = a·r + c) y **dos espirales**: fórmula cerrada y probada.
- **Haces n+1** (n paralelas + una suelta): regla de "reclamos" — cada paralela reclama clases
  enteras de H alrededor de su cruce con la suelta, los reclamos no se pisan, la suelta toma el
  resto. Comprobada sin suponerla en p = 29, 37, 41, 43 (`haces.c`).
- **Mezclas**: muchas soluciones son "unas filas de A y el resto de B"; en p=17 eso equivale a
  cambiar un bloque (clase de μ4 o μ8) de sitio.
- **Simetría**: casi todo lo ordenado tiene simetrías propias y se reconstruye desde pocas reinas
  semilla (`semillas.py`).

## Huecos concretos, por orden

1. **Regla del haz 2+2** (y 3+2, 3+3). La regla de clases enteras falla siempre. Hay miles de
   ejemplos en `data/haces_base_p{29,37,41}_2_2.txt` y `data/haces_base_p{19,31}_2_2.txt` (si falta
   alguno: `./haces_run.sh p 2 2 6`). Pista: la 2+2 de p=19 es una de dos espirales con 6 reinas
   cambiadas (mezcla 13+6). ¿Las 2+2 son siempre "dos espirales + un bloque movido"? ¿Qué reparto
   de filas tienen vistas desde los cuatro cruces?
2. **Mezcla = cambiar un bloque, en general.** Está comprobado fila a fila solo en p=17
   (`mezclas_porque.py`). Repetirlo en p=19 (tamaños 13+6, 10+9 frente a 15+4, 11+8) y convertirlo
   en un generador: partir de una solución, elegir un bloque que quepa, moverlo.
3. **Generador único.** Un programa que, dado un primo, produzca todas las ordenadas (una espiral,
   dos, haces n+1, las que salgan del punto 1 y 2, las de centro de simetría) sin búsqueda, y que
   se valide contra los recorridos exhaustivos de p = 13, 17, 19, 23.
4. **Terminar el censo**: 3+1 en p=41 y 2+2, 3+1 en p=43 quedaron sin terminar (`haces_run.sh`).
   Quitar duplicados (soluciones que admiten dos haces).
5. **La clase 5 ×128 de p=17** (índice 20 en el orden de `data/orbitas_p17.json` por (clase, tam)):
   única sin descripción en p=17. Sus coberturas usan pendientes casi emparejadas con su opuesta
   (−8,−5,−2,5,8): ¿es una simétrica con una recta "rota"?
6. Curiosidad sin explicar: nunca se pasa de una toroidal a otra moviendo 3 o 5 reinas (p=17, 19).

## Cosas ya descartadas (no repetir)

- Un solo "turno" con los dos cruces como polos (cociente de distancias): no ordena nada.
- Grado y número de términos del polinomio de la solución: no distingue.
- Encadenar mezclas de dos para llegar a las soluciones sin simetría: imposible (no son mezcla de
  dos de ningún par).
- Mezclas de tres con piezas complicadas: el azar acierta 30–50 %.

Guarda cada resultado en el README y en la memoria.
