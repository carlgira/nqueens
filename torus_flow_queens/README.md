# torus_flow_queens — un río sobre el toro que resuelve N-reinas

Una partícula viaja por la superficie de un toro arrastrada por un líquido.
La vuelta larga (θ) son las filas, el tubo (φ) son las columnas. Una vuelta
completa deja N reinas; si no se atacan, esa vuelta es una solución. La
siguiente vuelta es la siguiente solución.

- `core.py` — la EDO en tiempo continuo (`TorusFlow`) y `is_solution`.
- `core_pulsos.py` — primera versión: el campo cambiaba de golpe en cada reina.
- `experiments.py` — `chain`, `regimes`, `basins` (python experiments.py regimes 8).
- `index.html` — simulación en vivo (port JS de `core.py`) con toro 3D, mapa de
  calor, trazadores del líquido y la cadena de soluciones.

## El campo (continuo en el tiempo)

```
dr/dt = 1                          el río: una fila por unidad de tiempo
dx/dt = v,  dv/dt = (u − v)/τ      la partícula se deja llevar (τ = 0.12)
u     = κ |Z| (N/2π) sin(arg Z − φ),   Z = Σ_c e^{−E_c/T} e^{iφ_c}
E_c   = (1−σ) E_k(c) + σ E_{k+1}(c),   σ = smoothstep(p; 0.02, 0.3)
E_k(c)= H(fila k, c) + ε d(c, x + α·tiempo_al_centro)²
```

k es la fila recién cruzada y p la fracción de camino hacia la siguiente: el
objetivo del líquido pasa suavemente de una fila a la otra.

- **Reinas que nacen y se apagan.** Al cruzar el centro de una fila nace una
  reina en x, con intensidad 0. Sube entre edad 0.1 y 0.6, sus tres rayas
  (columna y dos diagonales) crecen hacia fuera a velocidad c = 3 filas por
  unidad de tiempo, y se apaga entre edad N−1 y N−0.55, justo antes de que la
  partícula vuelva a esa fila. El mapa de calor es continuo en t: las manchas
  frías y calientes se desplazan, no saltan.
- **Memoria** m(fila, c): se carga con un pulso gaussiano alrededor del centro
  de cada fila, repartido suavemente entre columnas vecinas, y decae a tasa λ por
  unidad de tiempo. Una celda recorrida en bucle satura en M* ≈ 1/(1 − e^{−λN}).
- **Costuras del tablero clásico.** Las diagonales no se envuelven: se apagan
  suavemente en la franja r ∈ [0, 0.35] ∪ [N−0.35, N), y en la textura queda una
  línea de falla en φ = 0. El tablero toroidal no tiene costuras.
- **α**: pendiente del río. Sin calor, la partícula traza la geodésica de
  pendiente α (la construcción mod-N, ver `wave_queens/` y la familia sturmiana).

La dinámica es determinista: la órbita, y por tanto la cadena de soluciones,
queda fijada por x₀, α y (β, λ, T).

## Lo que se ve (N = 8)

Un intento previo con potencial físico puro (partícula con inercia en un paisaje
de rayas) no funciona: en 1D no se pueden cruzar crestas y la partícula queda
pegada a su propia columna. Lo que funciona es que el líquido apunte al mínimo
*global* de la fila que viene (un min-conflicts continuo).

`python experiments.py regimes 8` (modelo continuo, 800 vueltas, x₀ = 0.3; β se
fija vía βM*, con M* = 1/(1 − e^{−λN})):

| λ \ βM* | 0.5 | 0.8 | 1.0 | 1.2 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| 0.005 | **fija** 99 % (1) | 57 % · 18 dist · rep 11 | 36 % · 19 | 28 % · 21 | 29 % · 26 | 29 % · 35 |
| 0.02 | fija 97 % (3) | fija 96 % (2) | 20 % · 33 | 17 % · 36 | 14 % · 33 | 9 % · 33 |
| 0.05 | fija 95 % (3) | **78 % · 9 dist · sin repetir** | fija 95 % (2) | 10 % · 26 | 5 % · 31 | 2 % · 12 |
| 0.15 | fija 97 % (2) | fija 99 % (2) | atascada | 6 % · 13 | fija 97 % (4) | 1 % · 6 |

(porcentaje = vueltas que son solución; "dist" = soluciones distintas; "rep k" =
la primera repetición ocurre en el eslabón k.)

- **βM* ≈ 0.5: se fija.** La memoria no basta para desestabilizar la solución
  encontrada; la órbita es periódica de periodo una vuelta.
- **βM* ≳ 1: encadena.** Recorre 20–36 soluciones distintas, con muchas vueltas
  de tránsito entre ellas. Repite al ritmo de un cumpleaños aleatorio (primera
  repetición hacia el eslabón 6–16; √(92·π/2) ≈ 12): la memoria es de celdas,
  no de soluciones enteras.
- **Frontera (βM* ≈ 0.8).** El régimen más parecido a "una solución tras otra":
  con λ = 0.05, 78 % de las vueltas son solución, cada una se sostiene ~70
  vueltas y en 800 vueltas no repite ninguna de las 9.

Comparado con la primera versión (`core_pulsos.py`, umbral nítido en βM* ≈ 1.42),
el modelo continuo baja el umbral a ≈ 1 y lo hace menos limpio (λ = 0.15 no es
monótono). La memoria continua se reparte entre columnas vecinas, así que pesa
menos por celda.

## Superficie o interior

Si el río fuera fijo (campo autónomo en T²), dos vueltas de la misma órbita no
podrían cruzarse (unicidad de la EDO). El mapa entre secciones de fila sería un
homeomorfismo del círculo, así que el **orden cíclico** de las columnas de todas
las vueltas se conservaría fila a fila. Una cadena S₁ → S₂ → … → S_m realizable en
la superficie necesita:

1. celdas disjuntas entre soluciones en cada fila, y
2. un orden cíclico de las m soluciones que sea el mismo en todas las filas,
   con el paso de vuelta actuando como una rotación de ese orden.

Con m ≤ 2 la condición 2 es vacía; desde m = 3 es fuerte. El máximo m = N lo dan
las traslaciones S + k de una solución toroidal (enlaza con el resultado
"toro perfecto sii solución toroidal"). Para encadenar soluciones arbitrarias
hace falta una dimensión más: la memoria del líquido (aquí) o meter la
partícula dentro del toro sólido, donde las vueltas pueden trenzarse.

## Abierto

- Medir cuántas soluciones alcanza cada x₀ (`experiments.py basins`) y si la
  unión de órbitas cubre las 92.
- Memoria de soluciones enteras (no solo celdas) para cadenas sin repetición.
- Comprobar la condición de orden cíclico: la cadena más larga de soluciones de
  N = 8 que un río fijo en la superficie puede recorrer.
