#Contact: Xingfu Wu

#MCS, ANL

#April 16, 2020

The package Benchmarks based on PolyBench 4.2 includes the following directories:

gemm    (gemm benchmark)

syr2k     (syr2k benchmark)

plopper  (code generation and compiling for both gemm and syr2k)

Before autotuning a benchmark, please check your clang compiler using gemm.bat and syr2k.bat under the corresponding directories to see if these benchmarks can be compiled sucessfully.

To autotune a benchmark, under the conda environment, use run.bat in the benchmark directory to test the autotuning framework. See the description of the framework for the details.

The run.bat looks like:

python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=20 --learner RF  (Note: autotuning the benchmark)

python findMin.py (Note: find the configuration for optimal performance from results.csv)


Installation requirements

This autotuning framework requires the following components: SOLLVE Clang with polly and pragma-clang-loop, configspace, scikit-optimize, autotune and ytopt. See the following installation instructions for the details.

1. Installing the latest version of SOLLVE

Use SOLLVE official site: https://github.com/SOLLVE/llvm-project.git to install Clang with polly, and use git checkout origin/pragma-clang-loop to install the pragma-clang-loop.

2. Installing the latest version of autotune and ytopt

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

If this works, this environment for the autotuning framework is ready to use. Please install scikit-learn==0.23.1 keras==2.3 because its latest version may cause some problem.
Please read the pdf file polybench-autotune.pdf for the details.
