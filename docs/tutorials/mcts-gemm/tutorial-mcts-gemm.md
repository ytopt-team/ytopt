Tutorial: Autotune tree space version of GEMM 
===================

This tutorial describes how to define autotuning problem and an evaluating method for autotuning [PolyBench](https://web.cse.ohio-state.edu/~pouchet.2/software/polybench/) GEMM kenel. 

We assume that you have checked out a copy of `ytopt`. For guidelines on how to get ytopt set up, refer [Install instructions](https://github.com/ytopt-team/ytopt/blob/tutorial/README.md) and [Install instructions for tree space](https://github.com/ytopt-team/ytopt/blob/mcts/ytopt/cmcts/README.md). 

Indentifying a problem to autotune 
-----------------------
In this tutorial, we target to autotune PolyBench GEMM kernel.

GEMM is a benchmark kernel for linear-algebra blas task; matrix-multiply C=alpha.A.B+beta.C [(reference)](https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1/tree/master/linear-algebra/blas/gemm). Save the related source and header files in the seprate folder: `gemm.c`, `gemm.h`, `polybench.c`, and `polybench.h`.

We omit presenting the files for space. For your convenience, we have the files in `<https://github.com/ytopt-team/ytopt/blob/mcts/ytopt/cmcts/benchmarks/gemm>`. 

Defining search space
-----------------------
We describe how to define search space.

Our search space conists of six loop trnsformations: 1) loop tiling, 2) loop interchange, 3) thread parallelization, 3) loop unrolling, 4) loop reversal, and 5) array packing. 

A search space generator is defined in [mctree_generator.py](https://github.com/ytopt-team/ytopt/blob/mcts/ytopt/cmcts/algorithms/mctree_generator.py)

<!-- --------------
First, we first define search space using ConfigSpace that is a python library `<https://automl.github.io/ConfigSpace/master/>`. -->

Customized Monte Carlo Tree Search Algorithm (MCTS)
-----------------------
We describe our customized MCTS:

A base MCTS algorithm that runs four steps of selection, expansion, simulation, and backpropagation is defined in [monte_carlo_tree_search_v1.py](https://github.com/ytopt-team/ytopt/blob/mcts/ytopt/cmcts/algorithms/monte_carlo_tree_search_v1.py)

The customized MCTS featueres with random walk, restart, and transfer learning. This is defined in [mctree_mcts.py](https://github.com/ytopt-team/ytopt/blob/mcts/ytopt/cmcts/mctree_mcts.py)

<!-- --------------
First, we first define search space using ConfigSpace that is a python library `<https://automl.github.io/ConfigSpace/master/>`. -->

Run the following
-----------------------
- Go to where `problem.py` such as

`
cd ytopt/cmcts/benchmarks/gemm
`
- Start search

`CLANG_PREFIX=/scratch/jkoo/sw/clang13/llvm-project/build/ ./autotune_mcts.sh`

Outputs look like
-----------------------

```

.....RUN mcts.....
n_run: 300 n_run_1: 50 n_run_2: 250 r_idx poly_v1_test
Extract loop nests...
$ (cd /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/base && /scratch/jkoo/sw/clang13/llvm-project/build//bin/clang /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm.c /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/polybench.c -I/scratch/jkoo/sw/clang13/llvm-project/build//projects/openmp/runtime/src -I/scratch/jkoo/sw/clang13/llvm-project/build//runtimes/runtimes-bins/openmp/runtime/src -L/scratch/jkoo/sw/clang13/llvm-project/build//runtimes/runtimes-bins/openmp/runtime/src -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native -I/home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm -DLARGE_DATASET -mllvm -polly-only-func=kernel_gemm -iquote /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm -iquote /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm -flegacy-pass-manager -ferror-limit=1 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-reschedule=0 -mllvm -polly-pattern-matching-based-opts=0 -DPOLYBENCH_TIME=1 -g -gcolumn-info -fopenmp -mllvm -polly-omp-backend=LLVM -mllvm -polly-scheduling=static -Werror=pass-failed -mllvm -polly-output-loopnest=loopnests.json -o /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/base/gemm)
Writing LoopNest to 'loopnests.json'.

Exit with code 0 in 0:00:02.826495
Exit with code 0 in 0:00:00.579431
Execution completed in 0:00:00.579431; polybench measurement: 0.450682
Exit with code 0 in 0:00:00.527145
Execution completed in 0:00:00.527145; polybench measurement: 0.428569
Exit with code 0 in 0:00:00.528090
Execution completed in 0:00:00.528090; polybench measurement: 0.429653
Exit with code 0 in 0:00:00.527671
Execution completed in 0:00:00.527671; polybench measurement: 0.435196
Exit with code 0 in 0:00:00.576873
Execution completed in 0:00:00.576873; polybench measurement: 0.434453
============================================================================================= 0
exploration_weight init.......................................: 0.1
[2, 3, 4, 1, 4, 4, 1, 3, 4, 2, 3, 1, 5, 5, 3, 2, 1, 1, 2, 4, 2, 3, 4, 1, 4, 1, 1, 1, 3, 1, 5, 5, 5, 2, 4, 5, 3, 2, 4, 5, 2, 5, 3, 2, 5, 2, 3, 4, 3, 5]
Root time 0.434453
=========init============== 0 0 1 terminal depth 5
Run next experiment in /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/experiment1
Experiment 1
Function kernel_gemm:
  #pragma clang loop(loop3,loop4,loop5) tile sizes(16,16,5) peel(rectangular) floor_ids(floor6,floor7,floor8) tile_ids(tile9,tile10,tile11)
  #pragma clang loop(floor6,floor7,floor8,tile9,tile10,tile11) tile sizes(3,32,16,5,4,2) peel(rectangular) floor_ids(floor12,floor13,floor14,floor15,floor16,floor17) tile_ids(tile18,tile19,tile20,tile21,tile22,tile23)
  #pragma clang loop(floor12,floor13,floor14,floor15,floor16,floor17,tile18,tile19,tile20,tile21,tile22,tile23) tile sizes(64,64,2,5,5,5,128,256,64,32,16,64) peel(rectangular) floor_ids(floor24,floor25,floor26,floor27,floor28,floor29,floor30,floor31,floor32,floor33,floor34,floor35) tile_ids(tile36,tile37,tile38,tile39,tile40,tile41,tile42,tile43,tile44,tile45,tile46,tile47)
  #pragma clang loop(floor24,floor25,floor26,floor27,floor28,floor29,floor30,floor31,floor32,floor33,floor34,floor35,tile36,tile37,tile38,tile39,tile40,tile41,tile42) tile sizes(3,32,16,8,128,2,2,2,3,2,2,8,64,2,32,2,2,8,16) floor_ids(floor48,floor49,floor50,floor51,floor52,floor53,floor54,floor55,floor56,floor57,floor58,floor59,floor60,floor61,floor62,floor63,floor64,floor65,floor66) tile_ids(tile67,tile68,tile69,tile70,tile71,tile72,tile73,tile74,tile75,tile76,tile77,tile78,tile79,tile80,tile81,tile82,tile83,tile84,tile85)
  #pragma clang loop(floor48,floor49,floor50,floor51,floor52,floor53,floor54,floor55,floor56,floor57,floor58,floor59,floor60,floor61,floor62,floor63,floor64,floor65,floor66) tile sizes(64,8,3,4,32,128,4,5,5,2,32,256,64,3,5,32,5,8,8) floor_ids(floor86,floor87,floor88,floor89,floor90,floor91,floor92,floor93,floor94,floor95,floor96,floor97,floor98,floor99,floor100,floor101,floor102,floor103,floor104) tile_ids(tile105,tile106,tile107,tile108,tile109,tile110,tile111,tile112,tile113,tile114,tile115,tile116,tile117,tile118,tile119,tile120,tile121,tile122,tile123)
$ (cd /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/experiment1 && /scratch/jkoo/sw/clang13/llvm-project/build//bin/clang /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/experiment1/gemm.c /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/experiment1/polybench.c -I/scratch/jkoo/sw/clang13/llvm-project/build//projects/openmp/runtime/src -I/scratch/jkoo/sw/clang13/llvm-project/build//runtimes/runtimes-bins/openmp/runtime/src -L/scratch/jkoo/sw/clang13/llvm-project/build//runtimes/runtimes-bins/openmp/runtime/src -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native -I/home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm -DLARGE_DATASET -mllvm -polly-only-func=kernel_gemm -iquote /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm -iquote /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm -flegacy-pass-manager -ferror-limit=1 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-reschedule=0 -mllvm -polly-pattern-matching-based-opts=0 -DPOLYBENCH_TIME=1 -fopenmp -mllvm -polly-omp-backend=LLVM -mllvm -polly-scheduling=static -Werror=pass-failed -o /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/experiment1/gemm)
Exit with code 1 in 0:00:15.424723
env.counter 1
best_depth: [inf, inf, inf, inf, inf] 2
=========init============== 0 1 2 terminal depth 3
Run next experiment in /home/jkoo/github/ytopt-mcts/ytune/ytopt/ytopt/search/mcts/benchmarks/gemm/gemm_mcts/mctree-frh3o5ab/experiment2
Experiment 2
 --> 20.577
```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
