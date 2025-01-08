# Sw4lite
**Sw4lite** is a bare bone version of [SW4](https://geodynamics.org/cig/software/sw4) ([Github](https://github.com/geodynamics/sw4)) intended for testing performance optimizations in a few
important numerical kernels of SW4.

To build
--------
The Makefiles are suited for our systems at LLNL and LBNL; you will have to modify them to suit your system.

Type:
```
make
```
to build the code with OpenMP. The executable will be named `optimize_mp_hostname/sw4lite`.

A debug version with OpenMP can be built by:
```
make debug=yes
```
which will be located at `debug_mp_hostname/sw4lite`.

To build with only C code (no Fortran) and with OpenMP, type:
```
make ckernel=yes
```
The executable will be `optimize_mp_c_hostname/sw4lite`.

To build without OpenMP type:
```
make openmp=no
```
The executable will be `optimize_hostname/sw4lite`.

The Cuda version is built by:
```
make -f Makefile.cuda
```
and the executable will be under `optimize_cuda_hostname/sw4lite`.

More options are described in the Makefile.

Experimental cmake build is available for cuda build:
```bash
mkdir build;
cd build;
cmake ..; # optionally add -DCMAKE_PREFIX_PATH=$PWD/../../lapack_build/ if lapack is not found by default.
make;
```

To run
------

To run sw4lite with OpenMP threading, you need to assign the number of threads per
MPI-task by setting the environment variable OMP_NUM_THREADS, e.g.,
```
setenv OMP_NUM_THREADS 4
```
An example input file is provided under `tests/pointsource/pointsource.in`. This case solves the
elastic wave equation for a single point source in a whole space or a half space. The input file is
given as argument to the executable, as in the example:
```
mpirun -np 16 sw4lite pointsource.in
```
Output from a run is provided at `tests/pointsource/pointsource.out`.
For this point source example, the analytical solution is known. The error is printed at the end:
```
Errors at time 0.6 Linf = 0.569416 L2 = 0.0245361 norm of solution = 3.7439
```
When modifying the code, it is important to verify that these numbers have not changed.

Some timings are also output. The average execution times (in seconds) over all MPI processes are reported as follows:
1. Total execution time for the time stepping loop,
2. Communication between MPI-tasks (BC comm)
3. Imposing boundary conditions (BC phys),
4. Evaluating the difference scheme for divergence of the stress tensor (Scheme),
5. Evaluating supergrid damping terms (Supergrid), and
6. Evaluating the forcing functions (Forcing)

The code under `tests/testil` is a stand alone single-core program that only exercises the computational kernel (Scheme).
