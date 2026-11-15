#!/bin/bash
# ./haces_run.sh p n m K : reparte ./haces en K procesos y junta en data/haces_base_p{p}_{n}_{m}.txt
p=$1; n=$2; m=$3; K=$4; s=$(date +%s)
for k in $(seq 0 $((K-1))); do ./haces $p $n $m $k $K > data/.haces_${p}_${n}_${m}_$k.part 2>/dev/null & done
wait
cat data/.haces_${p}_${n}_${m}_*.part > data/haces_base_p${p}_${n}_${m}.txt; rm data/.haces_${p}_${n}_${m}_*.part
echo "p=$p $n+$m: $(wc -l < data/haces_base_p${p}_${n}_${m}.txt) base, $(( $(date +%s) - s ))s"
