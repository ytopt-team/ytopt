#!/bin/bash
#COBALT -A Intel -n 4096 -t 30 -O runs4096x1x64 --attrs mcdram=cache:numa=quad

export OMP_NUM_THREADS=64

           aprun -n 4096 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out.txt
           aprun -n 4096 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out2.txt
           aprun -n 4096 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out3.txt
           aprun -n 4096 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out4.txt
           aprun -n 4096 -N 1 -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out5.txt


