Tutorial: Autotune the block matrix multiplication 
===================

This tutorial describes how to define autotuning problem and an evaluating method for autotuning the block matrix multiplication. 

We assume that you have checked out a copy of `ytopt`. For guidelines on how to get ytopt set up, refer [Install instructions](https://github.com/ytopt-team/ytopt/blob/tutorial/README.md). 

This example including the source code is borrowed from [http://opentuner.org/tutorial/gettingstarted/](http://opentuner.org/tutorial/gettingstarted/).

Indentifying a problem to autotune 
-----------------------
In this tutorial, we target to autotune the block size for matrix multiplication. Save the related source files in the seprate folder: `mmm_block.cpp`. We have the files in `<https://github.com/ytopt-team/ytopt/tree/tutorial/ytopt/benchmark/mmm-block/mmm_problem/mmm_block.cpp>`.


```python
#include <stdio.h>
#include <cstdlib>

#define N 100

int main(int argc, const char** argv)
{

  int n = BLOCK_SIZE * (N/BLOCK_SIZE);
  int a[N][N];
  int b[N][N];
  int c[N][N];
  int sum=0;
  for(int k1=0;k1<n;k1+=BLOCK_SIZE)
  {
      for(int j1=0;j1<n;j1+=BLOCK_SIZE)
      {
          for(int k1=0;k1<n;k1+=BLOCK_SIZE)
          {
              for(int i=0;i<n;i++)
              {
                  for(int j=j1;j<j1+BLOCK_SIZE;j++)
                  {
                      sum = c[i][j];
                      for(int k=k1;k<k1+BLOCK_SIZE;k++)
                      {               
                          sum += a[i][k] * b[k][j];
                      }
                      c[i][j] = sum;
                  }
              }
          }
      }
         }
  return 0;
}
```

Defining autotuning problem
-----------------------
We describe how to define your search problem `<https://github.com/ytopt-team/ytopt/blob/tutorial/ytopt/benchmark/mmm-block/mmm_problem/problem.py>`

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
```

Our search space contains one parameter; `BLOCK_SIZE`: number of blocks.  


```python
# create an object of ConfigSpace
cs = CS.ConfigurationSpace(seed=1234)
#block size for openmp dynamic schedule
p0= CSH.UniformIntegerHyperparameter(name='BLOCK_SIZE', lower=1, upper=10, default_value=5)
cs.add_hyperparameters([p0])
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
obj = Plopper(dir_path+'/mmm_block.cpp',dir_path)

x1=['BLOCK_SIZE']
def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]]]
        print('CONFIG:',point)
        params = ["BLOCK_SIZE"]
        result = obj.findRuntime(value, params)
        return result

    x = np.array([point['BLOCK_SIZE']])
    results = plopper_func(x)
    print('OUTPUT:%f',results)
    return results
```

The following describes our evaluating function, Plopper. You can find it `<https://github.com/ytopt-team/ytopt/blob/tutorial/ytopt/benchmark/mmm-block/plopper/plopper.py>`.  


```python
import os, sys, subprocess, random
random.seed(1234)

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
    
    def findRuntime(self, x, params):
        interimfile = ""
        exetime = 1
        
        # Generate intermediate file
        dictVal = self.createDict(x, params)

        #compile and find the execution time
        tmpbinary = self.outputdir + '/tmp.bin'
        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]
        gcc_cmd = 'g++ ' + kernel_dir +'/mmm_block.cpp '
        gcc_cmd += ' -D{0}={1}'.format('BLOCK_SIZE', dictVal['BLOCK_SIZE'])
        gcc_cmd += ' -o ' + tmpbinary
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

`findRuntime()` first calls `createDict()` to obatain configuration. 
After that, it generates the commandline `gcc_cmd` for compiling the modified source code and the commandline `run_cmd` for executing the compiled code. 
Then, it finds the compilation status using subprocess; finds the execution time of the compiled code; and returns the execution time as cost to the search module. 


```python
    def findRuntime(self, x, params):
        interimfile = ""
        exetime = 1
        
        # Generate intermediate file
        dictVal = self.createDict(x, params)

        #compile and find the execution time
        tmpbinary = self.outputdir + '/tmp.bin'
        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]
        gcc_cmd = 'g++ ' + kernel_dir +'/mmm_block.cpp '
        gcc_cmd += ' -D{0}={1}'.format('BLOCK_SIZE', dictVal['BLOCK_SIZE'])
        gcc_cmd += ' -o ' + tmpbinary
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

`python -m ytopt.search.ambs --evaluator ray --problem ytopt.benchmark.mmm-block.mmm_problem.problem.Problem --max-evals=5 --learner RF
`

--------------
Once autotuning kick off, ytopt.log, results.csv, and results.json will be rendered.

We can track the results of each run configuration from `ytopt.log` shows the following: 

```
2021-07-30 15:35:14|15364|INFO|ytopt.search.search:53] Created "ray" evaluator
2021-07-30 15:35:14|15364|INFO|ytopt.search.search:54] Evaluator: num_workers is 1
2021-07-30 15:35:14|15364|INFO|ytopt.search.hps.ambs:47] Initializing AMBS
2021-07-30 15:35:14|15364|INFO|ytopt.search.hps.optimizer.optimizer:51] Using skopt.Optimizer with RF base_estimator
2021-07-30 15:35:14|15364|INFO|ytopt.search.hps.ambs:79] Generating 1 initial points...
2021-07-30 15:35:15|15364|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'BLOCK_SIZE': '5'}
2021-07-30 15:35:17|15364|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"BLOCK_SIZE": "5"} --> 0.144
2021-07-30 15:35:17|15364|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'BLOCK_SIZE': '5'} y: 0.144
2021-07-30 15:35:17|15364|INFO|ytopt.search.hps.ambs:92] Refitting model with batch of 1 evals
2021-07-30 15:35:17|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:119] tell: {'BLOCK_SIZE': '5'} --> ('5',): evaluated objective: 0.144
2021-07-30 15:35:17|15364|INFO|ytopt.search.hps.ambs:94] Drawing 1 points with strategy cl_max
2021-07-30 15:35:18|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:84] _ask: ['6'] lie: 0.144
2021-07-30 15:35:18|15364|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'BLOCK_SIZE': '6'}
2021-07-30 15:35:19|15364|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"BLOCK_SIZE": "6"} --> 0.139
2021-07-30 15:35:19|15364|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'BLOCK_SIZE': '6'} y: 0.139
2021-07-30 15:35:19|15364|INFO|ytopt.search.hps.ambs:92] Refitting model with batch of 1 evals
2021-07-30 15:35:19|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:119] tell: {'BLOCK_SIZE': '6'} --> ('6',): evaluated objective: 0.139
2021-07-30 15:35:19|15364|INFO|ytopt.search.hps.ambs:94] Drawing 1 points with strategy cl_max
2021-07-30 15:35:19|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:84] _ask: ['2'] lie: 0.144
2021-07-30 15:35:19|15364|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'BLOCK_SIZE': '2'}
2021-07-30 15:35:21|15364|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"BLOCK_SIZE": "2"} --> 0.303
2021-07-30 15:35:21|15364|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'BLOCK_SIZE': '2'} y: 0.303
2021-07-30 15:35:21|15364|INFO|ytopt.search.hps.ambs:92] Refitting model with batch of 1 evals
2021-07-30 15:35:21|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:119] tell: {'BLOCK_SIZE': '2'} --> ('2',): evaluated objective: 0.303
2021-07-30 15:35:21|15364|INFO|ytopt.search.hps.ambs:94] Drawing 1 points with strategy cl_max
2021-07-30 15:35:21|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:84] _ask: ['8'] lie: 0.303
2021-07-30 15:35:21|15364|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'BLOCK_SIZE': '8'}
2021-07-30 15:35:23|15364|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"BLOCK_SIZE": "8"} --> 0.128
2021-07-30 15:35:23|15364|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'BLOCK_SIZE': '8'} y: 0.128
2021-07-30 15:35:23|15364|INFO|ytopt.search.hps.ambs:92] Refitting model with batch of 1 evals
2021-07-30 15:35:23|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:119] tell: {'BLOCK_SIZE': '8'} --> ('8',): evaluated objective: 0.128
2021-07-30 15:35:23|15364|INFO|ytopt.search.hps.ambs:94] Drawing 1 points with strategy cl_max
2021-07-30 15:35:23|15364|DEBUG|ytopt.search.hps.optimizer.optimizer:84] _ask: ['9'] lie: 0.303
2021-07-30 15:35:23|15364|INFO|ytopt.evaluator.evaluate:104] Submitted new eval of {'BLOCK_SIZE': '9'}
2021-07-30 15:35:25|15364|INFO|ytopt.search.hps.ambs:85] Elapsed time: 00:00:10.34
2021-07-30 15:35:25|15364|INFO|ytopt.evaluator.evaluate:206] New eval finished: {"BLOCK_SIZE": "9"} --> 0.125
2021-07-30 15:35:25|15364|INFO|ytopt.evaluator.evaluate:217] Requested eval x: {'BLOCK_SIZE': '9'} y: 0.125
2021-07-30 15:35:25|15364|INFO|ytopt.search.hps.ambs:101] Hyperopt driver finishing
```

Look up the best configuration (found so far) and its value by inspecting the following created file: `results.csv` and `results.json`. 

In this run, the best configuration and its runtime is obtained:

`{'BLOCK_SIZE': '9'}: 0.125`
