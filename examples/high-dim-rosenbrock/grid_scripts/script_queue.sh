#!/bin/bash

for x in `seq 2 2 10`;
do
    for y in `seq 2 2 10`;
    do
        sbatch -p cpulong /home/soldasim/BOLFI.jl/examples/high-dim/script.sh $x $y
    done
done
