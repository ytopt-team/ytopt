This directory includes the codes and scripts for using ytopt to autotune the XSBench Version 19, which is one of ECP Proxy Applications 
(https://github.com/ANL-CESAR/XSBench). It requires installing ytopt conda environment (https://github.com/ytopt-team/ytopt). For the MPI/OpenMP application, it requires MPI and OpenMP programming environments. You can test it at small scale, 
however, make sure to change the compilers to yours.

The list of folders:
```
Summit            Autotuning scripts and needed codes on Summit
Theta             Autotuning scripts and needed codes on Theta
xsbench-mpi.zip   The zip file including both folders Summit and Theta
xsbench-mixed.zip The zip file including Theta/xsbench-omp/tile-unroll which is the mixed code with the OpenMP pragmas and Clang loop optimization pragmas
```
