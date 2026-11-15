#!/bin/sh
# Repite todos los experimentos de caminos con el desempate uniforme (octubre 2026).
set -e
./grupos 13 t 2000 > data/grupos_p13_t.txt
for c in c o n; do ./grupos 17 $c 2000 > data/grupos_p17_$c.txt; done
./grupos 19 n 300 > data/grupos_p19_n.txt
for c in c o n; do for r in "tot 6" "tot 10" "sep 2" "quad 0"; do
  set -- $r; ./grupos2 17 $c $1 $2 300 60 > data/g2_p17_${c}_$1$2.txt; done; done
for r in "tot 6" "tot 10" "sep 2"; do set -- $r; ./grupos2 19 n $1 $2 150 60 > data/g2_p19_n_$1$2.txt; done
./grupos2 19 n quad 0 30 40 > data/g2_p19_n_quad0.txt
for r in "tot 6" "sep 2"; do set -- $r; ./grupos3 17 n $1 $2 150 40 > data/g3_p17_n_$1$2.txt; done
for r in "tot 6" "sep 2"; do set -- $r; ./grupos3 19 n $1 $2 40 40 > data/g3_p19_n_$1$2.txt; done
echo FIN
