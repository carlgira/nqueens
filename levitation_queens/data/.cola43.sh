#!/bin/bash
cd /Users/carlgira/files/nqueens/nqueens/levitation_queens
while pgrep -f "haces 41 3 1" > /dev/null; do sleep 20; done
./haces_run.sh 43 2 2 6 > data/haces_run_p43_2_2.log 2>&1
./haces_run.sh 43 3 1 6 > data/haces_run_p43_3_1.log 2>&1
