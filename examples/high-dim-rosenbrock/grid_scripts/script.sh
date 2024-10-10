#!/bin/sh

X_DIM=$1
Y_DIM=$2
NAME="grid_x${X_DIM}y${Y_DIM}"

DATE=$(date +"%Y-%m-%d")
ID="${DATE}-${RANDOM}"
OUT="/home/soldasim/BOLFI.jl/examples/high-dim/data/${NAME}-${ID}/log-${ID}.txt"

mkdir /home/soldasim/BOLFI.jl/examples/high-dim/data/${NAME}-${ID}

export JULIA_NUM_THREADS=1
julia /home/soldasim/BOLFI.jl/examples/high-dim/script.jl $ID $NAME $X_DIM $Y_DIM 1>$OUT 2>$OUT
