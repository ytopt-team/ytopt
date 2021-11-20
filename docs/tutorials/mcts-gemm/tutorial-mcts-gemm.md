Tutorial: Autotune tree space version of GEMM 
===================

This tutorial describes how to define autotuning problem and an evaluating method for autotuning PolyBench GEMM kenel. 

We assume that you have checked out a copy of `ytopt`. For guidelines on how to get ytopt set up, refer [Install instructions](https://github.com/ytopt-team/ytopt/blob/tutorial/README.md) and [Install instructions for tree space](https://github.com/ytopt-team/ytopt/blob/mcts/ytopt/search/mcts/README.md). 

Run the following
-----------------------
`CLANG_PREFIX=/scratch/jkoo/sw/clang13/llvm-project/build/ ./benchmarks/gemm/autotune_mcts.sh`

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

Indentifying a problem to autotune 
-----------------------
In this tutorial, we target to autotune ECP XSBench app `<https://github.com/ANL-CESAR/XSBench>`.

XSBench is a mini-app representing a key computational kernel of the Monte Carlo neutron transport algorithm [(reference)](https://github.com/ANL-CESAR/XSBench). Save the related source and header files in the seprate folder: `mmp.c`, `Main.c`, `Materials.c`, `XSutils.c`, `XSbench_header.h`, `make.bat`. 

We omit presenting the files for space. For your convenience, we have the files in `<https://github.com/ytopt-team/ytopt/tree/tutorial/ytopt/benchmark/xsbench-omp/xsbench>`. 

Defining autotuning problem
-----------------------
We describe how to define your search problem `<https://github.com/ytopt-team/ytopt/blob/tutorial/ytopt/benchmark/xsbench-omp/xsbench/problem.py>`

--------------
First, we first define search space using ConfigSpace that is a python library `<https://automl.github.io/ConfigSpace/master/>`.


```python
# import required library
import os, sys, time, json, math
import numpy as np
from autotune import TuningProblem
from autotune.space import *
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper
```

Our search space contains three parameters: 1) `p0`: number of threads, 2) `p1`: block size for openmp dynamic schedule, 3) `p2`: turn on/off omp parallel.  


```python
# create an object of ConfigSpace 
cs = CS.ConfigurationSpace(seed=1234)
# number of threads
p0= CSH.UniformIntegerHyperparameter(name='p0', lower=4, upper=8, default_value=8)
#block size for openmp dynamic schedule
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['10','20','40','64','80','100','128','160','200'], default_value='100')
#omp parallel
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma omp parallel for", " "], default_value=' ')
#add parameters to search space object
cs.add_hyperparameters([p0, p1, p2])
# problem space
input_space = cs
output_space = Space([Real(0.0, inf, name="time")])
```

--------------
Then, we need to define the objective function `myobj` to evaluate a point in the search space. 

In this example, we define an evaluating method (Plopper) for code generation and compilation. 
Plopper take source code and output directory and return an execution time. 


```python
dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp.c',dir_path)

x1=['p0','p1','p2']
def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]],point[x1[1]],point[x1[2]]]
        print('CONFIG:',point)
        params = ["P0","P1","P2"]
        result = obj.findRuntime(value, params)
        return result
    x = np.array([point[f'p{i}'] for i in range(len(point))])
    results = plopper_func(x)
    print('OUTPUT:%f',results)
    return results
```

The following describes our evaluating function, Plopper. You can find it `<https://github.com/ytopt-team/ytopt/blob/tutorial/ytopt/benchmark/xsbench-omp/plopper/plopper.py>`.  


```python
import os, sys, subprocess, random

class Plopper:
    def __init__(self,sourcefile,outputdir):
        self.sourcefile = sourcefile
        self.outputdir = outputdir+"/tmp_files"

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return(dictVal)

    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value != 'None': 
                            modify_line = modify_line.replace('#'+key, str(value))

                if modify_line != line:
                    f2.write(modify_line)
                else:
                    f2.write(line)     

    def findRuntime(self, x, params):
        interimfile = ""
        exetime = 1
        counter = random.randint(1, 10001)         
        interimfile = self.outputdir+"/tmp_"+str(counter)+".c"
        
        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        #compile and find the execution time
        tmpbinary = interimfile[:-2]
        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]
        gcc_cmd = "gcc -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 " + \
        " -o " + tmpbinary + " " + interimfile +" " + kernel_dir + "/Materials.c " \
        + kernel_dir + "/XSutils.c " + " -I" + kernel_dir + \
        " -lm" + " -L${CONDA_PREFIX}/lib"
        run_cmd = kernel_dir + "/exe.pl " + tmpbinary
        
        #Find the compilation status using subprocess
        compilation_status = subprocess.run(gcc_cmd, shell=True, stderr=subprocess.PIPE)

        #Find the execution time only when the compilation return code is zero, else return infinity
        if compilation_status.returncode == 0 :
            execution_status = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE)
            exetime = float(execution_status.stdout.decode('utf-8'))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime 
```

This file consists of several components.

`__init__()` takes paths of the source file and output directory, and creates the output directory if it does not exists.   


```python
def __init__(self,sourcefile,outputdir):
    # Initilizing global variables
    self.sourcefile = sourcefile
    self.outputdir = outputdir+"/tmp_files"

    if not os.path.exists(self.outputdir):
        os.makedirs(self.outputdir)
```

`createDict()` generates a dictionary for parameter labels and values.


```python
def createDict(self, x, params):
    dictVal = {}
    for p, v in zip(params, x):
        dictVal[p] = v
    return(dictVal)
```

`plotValues()` replaces the Markers in the source file with the corresponding prameter values of the parameter dictionary. 
For example, a sampled value for number of threads `p0` replaces `#P0` in line 349 `input.nthreads = #P0` of `mmp.c` that is the original source file. 


```python
def plotValues(self, dictVal, inputfile, outputfile):
    with open(inputfile, "r") as f1:
        buf = f1.readlines()
    with open(outputfile, "w") as f2:
        for line in buf:
            modify_line = line
            for key, value in dictVal.items():
                if key in modify_line:
                    if value != 'None': #For empty string options
                        modify_line = modify_line.replace('#'+key, str(value))
            if modify_line != line:
                f2.write(modify_line)
            else:
                f2.write(line)  #To avoid writing the Marker
```

`findRuntime()` first calls `createDict()` to obatain configuration values and `plotValues()` to modify the original source code. 
After that, it generates the commandline `gcc_cmd` for compiling the modified source code and the commandline `run_cmd` for executing the compiled code. 
Then, it finds the compilation status using subprocess; finds the execution time of the compiled code; and returns the execution time as cost to the search module. 


```python
def findRuntime(self, x, params):
    interimfile = ""
    exetime = 1
    counter = random.randint(1, 10001) # To reduce collision increasing the sampling intervals          
    interimfile = self.outputdir+"/tmp_"+str(counter)+".c"

    # Generate intermediate file
    dictVal = self.createDict(x, params)
    self.plotValues(dictVal, self.sourcefile, interimfile)

    #compile and find the execution time
    tmpbinary = interimfile[:-2]
    kernel_idx = self.sourcefile.rfind('/')
    kernel_dir = self.sourcefile[:kernel_idx]
    gcc_cmd = "gcc -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 " + \
    " -o " + tmpbinary + " " + interimfile +" " + kernel_dir + "/Materials.c " \
    + kernel_dir + "/XSutils.c " + " -I" + kernel_dir + \
    " -lm" + " -L${CONDA_PREFIX}/lib"
    run_cmd = kernel_dir + "/exe.pl " + tmpbinary

    #Find the compilation status using subprocess
    compilation_status = subprocess.run(gcc_cmd, shell=True, stderr=subprocess.PIPE)

    #Find the execution time only when the compilation return code is zero, else return infinity
    if compilation_status.returncode == 0 :
        execution_status = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE)
        exetime = float(execution_status.stdout.decode('utf-8'))
        if exetime == 0:
            exetime = 1
    else:
        print(compilation_status.stderr)
        print("compile failed")
    return exetime #return execution time as cost
```

Note: 
- For macOS it may need to compile it with `clang`. You can change `gcc` to `clang` such that `gcc_cmd = "clang -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 " + \`. 
- `exe.pl` computes average the execution time over multiple runs. We execute once in this example to save time. 

--------------
Last, we create an object of the autotuning problem. The problem will be called in the commandline implementation. 


```python
Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None)
```

Running and viewing Results
-----------------------

Now, we can run the following command to autotune our program: 
--evaluator flag sets which object used to evaluate models, --problem flag sets path to the Problem instance you want to use for the search, --max-evals flag sets the maximum number of evaluations, --learner flag sets the type of learner (surrogate model).

- Go to where `problem.py` such as

`
cd ytopt/benchmark/xsbench-omp/xsbench
`
- Start search

`python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=10 --learner RF
`

Note that use `python3` if your environment is built with python3. 

--------------
Once autotuning kick off, ytopt.log, results.csv, and results.json will be rendered.

We can track the results of each run configuration from `ytopt.log` shows the following (output lines are truncated for readability here): 

```
2021-07-28 15:51:49|2126|INFO|ytopt.search.search:53] Created "ray" evaluator
2021-07-28 15:51:49|2126|INFO|ytopt.search.search:54] Evaluator: num_workers is 1
2021-07-28 15:51:49|2126|INFO|ytopt.search.hps.ambs:47] Initializing AMBS
2021-07-28 15:51:49|2126|INFO|ytopt.search.hps.optimizer.optimizer:51] Using skopt.Optimizer with RF base_estimator
2021-07-28 15:51:49|2126|INFO|ytopt.search.hps.ambs:79] Generating 1 initial points...
2021-07-28 15:51:50|2126|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'p0': '6', 'p1': '200', 'p2': ' '}
2021-07-28 15:52:12|2126|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"p0": "6", "p1": "200", "p2": " "} --> 20.158
2021-07-28 15:52:12|2126|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'p0': '6', 'p1': '200', 'p2': ' '} y: 20.158
2021-07-28 15:52:12|2126|INFO|ytopt.search.hps.ambs:92] Refitting model with batch of 1 evals
2021-07-28 15:52:12|2126|DEBUG|ytopt.search.hps.optimizer.optimizer:119] tell: {'p0': '6', 'p1': '200', 'p2': ' '} --> ('6', '200', ' '): evaluated objective: 20.158
2021-07-28 15:52:13|2126|INFO|ytopt.search.hps.ambs:94] Drawing 1 points with strategy cl_max
2021-07-28 15:52:13|2126|DEBUG|ytopt.search.hps.optimizer.optimizer:84] _ask: ['7', '40', ' '] lie: 20.158
2021-07-28 15:52:13|2126|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'p0': '7', 'p1': '40', 'p2': ' '}
2021-07-28 15:52:36|2126|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"p0": "7", "p1": "40", "p2": " "} --> 21.687
2021-07-28 15:52:36|2126|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'p0': '7', 'p1': '40', 'p2': ' '} y: 21.687
2021-07-28 15:52:36|2126|INFO|ytopt.search.hps.ambs:92] Refitting model with batch of 1 evals
2021-07-28 15:52:36|2126|DEBUG|ytopt.search.hps.optimizer.optimizer:119] tell: {'p0': '7', 'p1': '40', 'p2': ' '} --> ('7', '40', ' '): evaluated objective: 21.687
2021-07-28 15:52:37|2126|INFO|ytopt.search.hps.ambs:94] Drawing 1 points with strategy cl_max
2021-07-28 15:52:37|2126|DEBUG|ytopt.search.hps.optimizer.optimizer:84] _ask: ['7', '200', '#pragma omp parallel for'] lie: 21.687
2021-07-28 15:52:37|2126|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'p0': '7', 'p1': '200', 'p2': '#pragma omp parallel for'}
2021-07-28 15:52:58|2126|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"p0": "7", "p1": "200", "p2": "#pragma omp parallel for"} --> 20.393
2021-07-28 15:52:58|2126|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'p0': '7', 'p1': '200', 'p2': '#pragma omp parallel for'} y: 20.393
2021-07-28 15:52:58|2126|INFO|ytopt.search.hps.ambs:92] Refitting model with batch of 1 evals
2021-07-28 15:52:58|2126|DEBUG|ytopt.search.hps.optimizer.optimizer:119] tell: {'p0': '7', 'p1': '200', 'p2': '#pragma omp parallel for'} --> ('7', '200', '#pragma omp parallel for'): evaluated objective: 20.393
2021-07-28 15:52:59|2126|INFO|ytopt.search.hps.ambs:94] Drawing 1 points with strategy cl_max
2021-07-28 15:52:59|2126|DEBUG|ytopt.search.hps.optimizer.optimizer:84] _ask: ['8', '100', ' '] lie: 21.687
2021-07-28 15:52:59|2126|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'p0': '8', 'p1': '100', 'p2': ' '}
2021-07-28 15:53:20|2126|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"p0": "8", "p1": "100", "p2": " "} --> 20.577
```

Look up the best configuration (found so far) and its value by inspecting the following created file: `results.csv` and `results.json`. 

In this run, the best configuration and its runtime is obtained:

`{"p0": "8", "p1": "200", "p2": "#pragma omp parallel for"}: 19.604`
