In this experiment, we use OpenMP version of XSBench to add more OpenMP pragmas and clang loop optimization pragmas such as unrolling and tiling 
into the source code so that we use ytopt autotuning framework to autotune the mixed OpenMP loop pragmas and clang loop pragmas version of XSBench 
to achieve the best performance. What we did is to integrate the new openmp pragmas with clang loop pragmas as parameters to autotune the XSBench (OpenMP version) and to 
make sure that the result is valid (verified). 

# Install the latest ytopt
Please follow the instruction from readme.md under Benchmarks to install the latest ytopt packages

# Install the latest LLVM clang/polly/openmp
Please follow the instruction from readme.md under Benchmarks to install the latest LLVM clang/polly/openmp.

# Contents
This directory includes two folders:

xsbench: the original codes from https://github.com/ANL-CESAR/XSBench; the codes with additional openmp pragmas and clang loop pragmas.

plopper: the Python script for the function plopper (which entails replacing the parameters with values and compiles and executes each configureation).

# Contact Information
Xingfu Wu, MCS, ANL.

April 12, 2021
