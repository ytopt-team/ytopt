#!/bin/bash
#COBALT -n 8 -t 60 -O runs8  -qdebug-flat-quad -A EE-ECP 

source bashrc.theta

aprun -n 8 -N 1 -cc depth -d 64 -j 1 tmp_files/7884 3 512 
