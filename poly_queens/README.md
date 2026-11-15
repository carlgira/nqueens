# poly_queens — geometria de polinomios

Continuacion del ultimo parrafo de `diag_waves/README.md`. Alli quedaron dos cosas
abiertas: que `Q(N)` es un permanente restringido, y que el 30% que el nivel de pares
no captura "necesita el desarrollo completo de cumulos".

**Resumen en una frase**: ese desarrollo **no existe**. La serie de cumulos tiene
radio de convergencia `~4.9/N` alrededor de `z=1`, asi que `z=0` —donde vive `Q(N)`—
queda fuera para todo `N>=6` y la serie diverge. No falta un termino: falta
convergencia.

---

## 1 · La factorizacion de Vandermonde · `vandermonde.py`

```
prod_{i<j} [ (s_i - s_j)^2 - (i-j)^2 ]  =  V(sigma - id) * V(sigma + id)
```

Es una **identidad de polinomios**, valida para todo N y para cualquier vector `s`, y se
prueba en una linea por diferencia de cuadrados:

```
(s_i-s_j)^2 - (i-j)^2 = [(s_i-i) - (s_j-j)] * [(s_i+i) - (s_j+j)]
```

Al multiplicar sobre `i<j`, el primer factor recorre el Vandermonde de `d-` y el segundo
el de `d+`. No hay induccion ni caso base. Luego:

> **sigma es solucion  <=>  V(sigma+id) · V(sigma-id) != 0**

`Q(N)` es el numero de puntos de la orbita de `S_N` que evitan la union de dos
discriminantes trasladados.

Los tres controles del modulo son cosas distintas y conviene no confundirlos:

| control | que comprueba | alcance | que lo limita |
|---|---|---|---|
| `check_identity` | la identidad en si, en puntos enteros y racionales arbitrarios | **N=160** (479 852 bits) | nada relevante; es `O(N^2)` |
| `check_equiv` | `delta != 0 <=> c = 0`, en permutaciones al azar y en soluciones mod-N | **N=121** | nada relevante |
| `check_count` | que los no-nulos son **exactamente** `Q(N)` | **N=9** (y N=12 por la via de `conflict_poly`) | recorrer las `N!` permutaciones |

O sea: el techo de 9 es del **conteo exhaustivo**, no de la identidad. Los no-nulos dan
2, 10, 4, 40, 92, 352 y, contando conflictos en vez de multiplicar Vandermondes,
724, 2680, 14200.

Dos cosas se caen de aqui:

**Un potencial donde la solucion es un maximo.** `log|V(d+)| + log|V(d-)|` es menos la
energia de Coulomb de dos gases 1D. `newton_queens/` y `wave_queens/` chocaron con que
"el estado difuminado radia menos que cualquier solucion — la solucion es un vertice de
energia alta, no un minimo". Con el log-Vandermonde eso se invierte: la barrera en el
conflicto es infinita y el estado difuminado vale `-inf`. Es el unico funcional del
proyecto entero en el que las soluciones son maximos genuinos.

**El metodo polinomico falla por un factor 3/2, medible.** El Nullstellensatz
combinatorio pide `grado = sum_i k_i` sobre la rejilla `[N]^N`:

```
deg V(x)·V(x+id)·V(x-id) = 3·N(N-1)/2        sum_i k_i = N(N-1)
```

Sobra `N(N-1)/2`. Dicho de otro modo: la rejilla paga **dos** familias de
transversales y el problema tiene **tres** (columnas, `d+`, `d-`). Es el mismo umbral
"2 facil, 3 dificil" que separa emparejamiento bipartito de 3-dimensional matching, y
que explica la NP-completitud de Gent-Jefferson-Nightingale desde el lado de los
polinomios.

---

## 2 · El polinomio de conflictos · `conflict_poly.py`

```
C_N(z) = sum_{sigma en S_N} z^{c(sigma)}       c = pares en conflicto (el testigo)

C_N(0) = Q(N)          C_N(1) = N!
```

Una funcion de particion con `z` de fugacidad; `Q(N)` es su temperatura cero. Todo el
problema es ir de `z=1` (donde sabemos contar) a `z=0` (donde queremos). Coeficientes
exactos por enumeracion en bloques, hasta N=12 (479M permutaciones, ~3.5 min).

Controles: `C_N(0) = Q(N)` exacto para N=4..12, y `E[C] = (2N-1)/3` exacto.

### El desarrollo de cumulos, en su sitio

```
log C_N(z)/N! = sum_k kappa_[k] (z-1)^k / k!          (cumulos factoriales)
log Q(N)/N!   = sum_k kappa_[k] (-1)^k / k!           evaluado en z-1 = -1
```

El **nivel 2 de `delsarte.py` es el primer termino solo**: `kappa_[1] = E[C]`. Ahora se
puede pedir el siguiente, y el siguiente.

| N | t1 | t2 | t3 | t4 | t5 | captura k=1 | k=2 | **k=3** | k=4 | k=5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | −6.333 | +0.081 | −1.595 | +1.283 | −3.198 | 74.3% | 73.4% | **92.1%** | 77.0% | 114.6% |
| 11 | −7.000 | +0.077 | −1.732 | +1.312 | −3.340 | 72.9% | 72.1% | **90.1%** | 76.4% | 111.2% |
| 12 | −7.667 | +0.087 | −1.913 | +1.457 | −3.738 | 73.5% | 72.7% | **91.1%** | 77.1% | 112.9% |

Dos lecturas:

- **El nivel de pares (k=2) no aporta nada**: `t2 = +0.08`, y con el signo equivocado.
  Es exactamente la sobredispersion medida en el README anterior
  (`kappa_[2] = Var[C] - E[C]`, y `Var/E = 1.026`). El 30% que faltaba **no** estaba en
  la correlacion de segundo orden.
- **El nivel de TRIPLES si**: `t3` es O(N) y lleva la captura de 70.6% a ~90%. La
  constante NO se puede leer en N<=12 — ahi vale −0.159 y sigue subiendo. Con los
  cumulos combinatorios de la seccion 3, hasta N=31, converge a **−0.1705**. Ver la
  tabla completa alli.

Y luego se rompe: k=4 vuelve a 77%, k=5 se pasa a 113%, y a partir de ahi los terminos
explotan (en N=12 el termino 20 vale 1.26·10^6 para una respuesta de −10.43).

### Por que se rompe: los ceros

El radio de convergencia de esa serie es la distancia de `z=1` al cero mas cercano de
`C_N`. Medido:

| N | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|
| `R = min\|r−1\|` | 1.072 | 0.845 | 0.713 | 0.612 | 0.543 | 0.488 | 0.446 | 0.412 |
| `N·R` | 5.36 | 5.07 | 4.99 | 4.90 | 4.88 | 4.88 | 4.91 | 4.94 |
| `z=0` | dentro | **fuera** | fuera | fuera | fuera | fuera | fuera | fuera |

```
R(N) ~ 4.9 / N
```

`N·R` se mantiene entre 4.88 y 5.07 en todo el rango. (Ajustando el exponente
localmente sale entre 0.93 y 1.14, asi que la ley es empirica sobre N=5..12, no
demostrada.)

> **El disco de convergencia se cierra sobre `z=1` como `1/N`. `z=0` queda fuera para
> todo `N>=6`. Ninguna truncacion de cumulos de orden fijo converge a `Q(N)`, y la
> serie completa tampoco.**

Es la version Lee-Yang de la pared del proyecto: no es falta de ingenio, es una
singularidad entre el punto donde sabemos contar y el punto donde queremos contar.

→ `fig_zeros.png`

### La reparametrizacion: probabilidades de conflicto complejas

Poniendo `p = 1/(1-r)` para cada cero `r`:

```
log Q(N)/N! = sum_i log|1 - p_i|          sum_i p_i = E[C] = (2N-1)/3   (exacto)
```

Verificado a precision maquina para N=5..12. Si todas las `p` fueran reales en `(0,1)`,
`C` seria una suma de Bernoullis **independientes** y valdria `Var[C] <= E[C]`. Como el
README anterior midio `Var/E = 1.026 > 1`, esa medida **prueba** que `C_N` no tiene
todas sus raices reales — un numero suyo se convierte en teorema. Contado: de 55 raices
en N=11, solo 3 son reales.

> `Q(N) = N!` por un producto de `C(N,2)` factores `(1-p_i)` con `p_i` **complejas**.
> La "correlacion entre pares" del README es literalmente la parte imaginaria de esas
> probabilidades.

La masa real esta clavada en `(2N-1)/3` —lineal, y es todo lo que ve el nivel 2— y la
imaginaria crece mucho mas rapido: `sum |Im p|` va 6.46, 9.81, 13.45, 17.96, 23.07,
29.05, 35.99 para N=6..12. Superlineal y compatible con `N^2`, aunque el cociente
`sum|Im p| / N^2` todavia sube (0.179 -> 0.250 en ese rango) y no se ha estabilizado:
con N<=12 no se puede fijar el exponente.

---

## 3 · Los cumulos sin permutaciones · `cumulants.py`

La seccion 2 muere en N=12 porque necesita las `N!` permutaciones. Pero para la
escalera solo hacen falta los primeros cumulos, y esos se cuentan por otro lado.

Un conflicto es una **restriccion lineal entre dos filas**:

```
d+ :  i + sigma(i) = j + sigma(j)   <=>   sigma(i) - sigma(j) = +(j-i)
d- :  sigma(i) - i = sigma(j) - j   <=>   sigma(i) - sigma(j) = -(j-i)
```

Hay `M = N(N-1)` de estas ("atomos"), `C` cuenta cuantas se cumplen, y

```
E[(C)_k] / k!  =  sum_{|S|=k} P(S)
```

Para un `S` dado las restricciones forman un grafo sobre las filas; cada componente
conexa queda fijada por **un** valor libre `t_p`, con `sigma(r) = t_p + o_p(r)`. Luego

```
P(S) = D(S) * (N-n)! / N!
```

con `n` las filas tocadas y `D(S)` el numero de tuplas `(t_1..t_m)` que dejan todos los
valores dentro de `[0,N)` y distintos. `D(S)` sale por inclusion-exclusion sobre las
colisiones entre componentes: cada colision fija una diferencia `t_p - t_q`, y la tupla
cae en una interseccion de intervalos.

Control a mano de `k=1`: un atomo de hueco `g` da `D = N-g`, `n = 2`, y
`sum_S P = 2 sum_g (N-g)^2 / (N(N-1)) = (2N-1)/3`. Es el `E[C]` del README anterior,
ahora derivado en vez de medido.

**Lo que lo hace viable**: `D(S)` depende solo de la *firma* de las componentes (sus
offsets), no de que filas concretas sean. Hay `O(N^3)` firmas frente a `O(N^6)`
subconjuntos, asi que memoizarla da un factor ~10. El bucle interno no toca `Fraction`:
acumula `D` en enteros separados por numero de filas y divide una sola vez.

**Control cruzado**: los tres cumulos coinciden **exactamente** (como racionales) con
los que da la enumeracion de las `N!` permutaciones, para N=4..12.

| N | 4 | 8 | 12 |
|---|---|---|---|
| `kappa_[1]` | 7/3 | 5 | 23/3 |
| `kappa_[2]` | 5/9 | 17/105 | 257/1485 |
| `kappa_[3]` | −16/27 | 55/7 | 79553/6930 |

### La escalera, ahora hasta N=26 · `ladder3.py`

Con `kappa_[1..3]` exactos y `Q(N)` conocido (A000170, N<=27), la captura de cada nivel
se mide en todo el rango, no solo en N=10..12:

| N | 12 | 14 | 16 | 18 | 20 | 22 | 24 | 26 | asintotico |
|---|---|---|---|---|---|---|---|---|---|
| nivel 2 (pares) | 73.5% | 72.7% | 73.0% | 72.6% | 72.4% | 72.2% | 72.1% | 72.0% | **70.6%** |
| nivel 3 (ternas) | 91.1% | 90.1% | 90.5% | 90.0% | 89.9% | 89.7% | 89.6% | 89.4% | **~88.7%** |

El nivel 2 baja hacia su valor asintotico conocido (70.6%) y el nivel 3 lo acompana.

### La constante del nivel 3

`kappa_[3]` exacto para N=6..31 (26 valores). La constante es `t3/N = -kappa_[3]/(6N)`,
y converge despacio — hay una correccion `O(1/N)` grande:

| N | 12 | 16 | 20 | 24 | 28 | 31 |
|---|---|---|---|---|---|---|
| `t3/N` | −0.1594 | −0.1614 | −0.1630 | −0.1641 | −0.1650 | −0.1655 |
| Richardson `(N, N−2)` | −0.1592 | −0.1676 | −0.1697 | −0.1700 | −0.1704 | −0.1706 |

La extrapolacion de Richardson —que es exactamente el ajuste `c + a/N`— se estabiliza:
−0.17049, −0.17036, −0.17046, −0.17038, −0.17067, −0.17044, −0.17065 en N=23..31.

```
t3/N  ->  -0.1705 +- 0.0005
```

Aplicar Richardson una segunda vez NO afina mas (sale entre −0.1701 y −0.1726, mas
disperso que la entrada): la correccion siguiente no es un `1/N^2` limpio. Y ajustar
`c + a/N + b/N^2` por minimos cuadrados sobre 6 puntos vecinos da −0.1727, con `b`
grande y negativo — esta mal condicionado, no es una medida mejor.

De ahi `alpha_nivel3 = 1.667 + 0.1705 = 1.8372` contra `alpha_real = 1.944`, o sea
`(1.8372-1)/(1.944-1) = 88.7%`, coherente con el 89.4% medido en N=26 y todavia bajando.

**Con N<=12 esto no se veia**: alli `t3/N` vale −0.159 y aun esta subiendo. Extender de
N=12 a N=31 mueve la constante un 7%.

### La forma cerrada de `kappa_[3]`: no hay por este camino, y esta acotado por que

`kappa_[1] = (2N-1)/3` es exacto y elemental. El objetivo era el equivalente para
`kappa_[3]`. El ansatz natural sale de la recurrencia
`kappa_[3]/6 = G3 - G1*G2 + G1^3/3`: los denominadores son `(N)_6`, `(N)_4` y un 27,
asi que

```
v(N) = kappa_[3] * 27 * N(N-1)(N-2)(N-3)(N-4)(N-5)
```

deberia ser un polinomio de grado 7. **Es entero para todo N medido, pero no es
polinomio.** Con los 26 valores exactos:

- diferencias finitas sobre los 12 N mayores: **ninguna se anula**, y a partir del
  orden 4 vuelven a crecer;
- separando por paridad (13 puntos por clase): **tampoco**. No es cuasi-polinomica de
  periodo 2.

Ajustar sin validar da basura, y se ve: grado 7 sobre los 8 mayores de cada paridad da
lideres 0.752 y 2.364, incompatibles entre si y con el −0.1705 medido. Es la "trampa
metodologica" del README anterior en su version de ajuste: con tantos grados de libertad
como puntos no estas ajustando, estas leyendo la respuesta.

Lo que queda: el periodo podria ser 6 —encajaria con el `gcd(N,6)=1` que aparece por
todo el proyecto— pero comprobarlo pide ~8 puntos por clase de resto, o sea N~60, y el
coste es `C(N(N-1), 3) ~ N^6/6`. Fuera de alcance por este camino.

---

## Lo que esto cierra y lo que abre

**Cierra**: la pregunta "cual es el nivel 3". Respuesta: existe, vale ~91%, y es el
ultimo. El nivel 2 no aporta nada y del 4 en adelante la serie diverge.

**Abre**, por orden de valor:

1. **La ley `4.9/N` con mas N.** No hace falta el polinomio entero: `R` se puede sacar
   del crecimiento `|kappa_[k]/k!|^{1/k}`, pero eso pide cumulos de orden alto y el
   coste de la seccion 3 es `C(M,k)` con `M = N(N-1)`: viable para k=3, no para k=15.
2. **Capacidad de Gurvits** sobre `prod_i (sum_j v_j s_{i+j} t_{i-j})`. Ojo: esa forma
   es **cubica**, no lineal, asi que no es establemente real y la teoria de van der
   Waerden no aplica tal cual. Verificar si falla la log-concavidad, y **cuanto** falla
   (signatura del hessiano), seria el analogo multivariante de lo que aqui se midio en
   una variable.

Y lo que **no** hay que volver a intentar: sumar niveles. Estan contados.

---

## Ejecutar

```bash
venv/bin/python -m poly_queens.vandermonde        # la identidad y Q(N) como no-anulacion
venv/bin/python -m poly_queens.conflict_poly 11   # ceros, escalera y divergencia
venv/bin/python -m poly_queens.plots              # fig_zeros.png
venv/bin/python -m poly_queens.cumulants 20      # cumulos sin enumerar permutaciones
venv/bin/python -m poly_queens.ladder3 26        # la escalera contra Q(N) real
```

`cumulants.py` cuesta `C(N(N-1),3)`: N=20 en ~40 s, N=26 en ~3 min, N=31 en ~11 min.
Los valores ya calculados hasta N=26 estan cacheados en `kappa.json`.

`conflict_poly 12` anade N=12 (~3.5 min). Los histogramas quedan cacheados en
`hist_N.npy`.
