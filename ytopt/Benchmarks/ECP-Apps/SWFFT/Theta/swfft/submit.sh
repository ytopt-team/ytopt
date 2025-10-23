#! /bin/bash

qsub --mode script -t 30 -q debug-cache-quad -A EE-ECP ./run.sh
