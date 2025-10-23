#!/bin/bash
#COBALT -n 8

#source bashrc.theta
module load cray-libsci
module load cray-fftw

nrep=3
ng=640

# CHAINING_MESH_THREADS in indat.params should equal RPN
ranks_per_node=1
threads_per_rank=64

exe="./TestDfft"

nodes=$COBALT_JOBSIZE
total_ranks=$((nodes*ranks_per_node))
cores_per_node=64
threads_per_core=$((ranks_per_node*threads_per_rank/cores_per_node))

#export KMP_AFFINITY=none
export OMP_NUM_THREADS=$threads_per_rank

aprun -n $total_ranks -N $ranks_per_node -cc depth -d $threads_per_rank -j $threads_per_core -e OMP_NUM_THREADS=$OMP_NUM_THREADS $exe $nrep $ng
