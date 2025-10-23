#!/bin/bash
#COBALT -A EE-ECP -n 1 -t 60 -O runs1x1x64 -qdebug-cache-quad  --env JOBID=

module unload darshan
module load intel
module load geopm
export OMP_NUM_THREADS=64

           aprun -n 1 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 1 1 1 -n 100 100 100 > out.txt
           aprun -n 1 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 1 1 1 -n 100 100 100 > out2.txt
           aprun -n 1 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 1 1 1 -n 100 100 100 > out3.txt
           aprun -n 1 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 1 1 1 -n 100 100 100 > out4.txt
           aprun -n 1 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 1 1 1 -n 100 100 100 > out5.txt


