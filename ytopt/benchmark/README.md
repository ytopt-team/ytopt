#Contact: Xingfu Wu

#MCS, ANL

#July 15, 2021

The benchamrks include the following directories:

dl (deep learning mnist)

xsbench-omp (openmp version of XSBench https://github.com/ANL-CESAR/XSBench)

xsbench-mpi-omp (hybrid MPI/OpenMP version of XSBench https://github.com/ANL-CESAR/XSBench)

To autotune a benchmark, under the conda environment, use run.bat in the benchmark directory to test the autotuning framework. See the description of the framework in the pdf file CCPE2021.pdf for the details.

The run.bat looks like:
```
python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=20 --learner RF  (Note: autotuning the benchmark)
python findMin.py (Note: find the configuration for optimal performance from results.csv)
```
- Note: The Benchmarks in this directory do not require the installation of LLVM Clang/Polly. For the benchmarks using Clang loop optimization pragmas, please look at the detailed instructions from the link https://github.com/ytopt-team/autotune/tree/master/Benchmarks
