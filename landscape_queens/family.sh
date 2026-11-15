#!/bin/sh
# J = [dc==0] + [dc==dr] - g * (dc/(N-1))^p   (ataque + "lejos es mejor" en columnas)
N=$1; g=$2; p=$3
awk -v N=$N -v g=$g -v p=$p 'BEGIN{for(dr=1;dr<N;dr++){for(dc=0;dc<N;dc++){j=(dc==0||dc==dr)?1:0; j-=g*(dc/(N-1))^p; printf "%.6f ",j} print ""}}' > Jf_$N.txt
./kernel $N file Jf_$N.txt | tail -1 | sed "s/^/g=$g p=$p /"
