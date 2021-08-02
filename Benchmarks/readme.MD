#Contact: Xingfu Wu

#MCS, ANL

#July 15, 2021

The package Benchmarks includes the following directories:

DL    (deep learning mnist)

xsbench-omp (openmp version of XSBench https://github.com/ANL-CESAR/XSBench)

xsbench-mpi-omp (hybrid MPI/OpenMP version of XSBench https://github.com/ANL-CESAR/XSBench)

...

To autotune a benchmark, under the conda environment, use run.bat in the benchmark directory to test the autotuning framework. See the description of the framework in the pdf file CCPE2021.pdf for the details.

The run.bat looks like:

python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=20 --learner RF  (Note: autotuning the benchmark)

python findMin.py (Note: find the configuration for optimal performance from results.csv)


Installation requirements:

This autotuning framework requires the following components: configspace, scikit-optimize, autotune and ytopt. See the following installation instructions for the details.


- Installing the latest version of autotune and ytopt

For the autotune and ytopt using ConfigSpace, the installation is as follows:

1)      Create a conda env ytopt
        conda create -n ytopt python=3.7
        conda activate ytopt

2)      pip install ConfigSpace

3)      git clone https://github.com/pbalapra/scikit-optimize.git
        cd scikit-optimize
        pip install -e  .

4)      git clone  https://github.com/ytopt-team/autotune.git
        cd autotune/
        git checkout version1
        pip install -e .

5)      git clone https://github.com/ytopt-team/ytopt.git
        cd ytopt/
        pip install -e .

Under the current directory, test the benchmark example:

python -m ytopt.search.ambs --evaluator ray --problem ytopt.benchmark.loopopt.problem.Problem

If this works, this environment for the autotuning framework is ready to use. Please install scikit-learn==0.23.1 because its latest version may cause some problem.


- Note: The Benchmarks in this directory do not require the installation of LLVM Clang/Polly. For the benchmarks using Clang loop optimization pragmas, please look at the detailed instructions from the link https://github.com/ytopt-team/autotune/tree/master/Benchmarks
