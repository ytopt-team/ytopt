Tutorial: Autotune the OpenMP version of XSBench 
===================

This tutorial describes how to define autotuning problem and an evaluating method for autotuning ECP XSBench app. 

We assume that you have checked out a copy of `ytopt`. For guidelines on how to get ytopt set up, refer [Install instructions](https://github.com/ytopt-team/ytopt/blob/tutorial/README.md). 

You can install openmpi openmpi-mpicc openmp for this example: `conda install -c conda-forge openmp openmpi openmpi-mpicc`

Indentifying a problem to autotune 
-----------------------
In this tutorial, we target to autotune ECP XSBench app `<https://github.com/ANL-CESAR/XSBench>`.

XSBench is a mini-app representing a key computational kernel of the Monte Carlo neutron transport algorithm [(reference)](https://github.com/ANL-CESAR/XSBench). Save the related source and header files in the seprate folder: `mmp.c`, `Main.c`, `Materials.c`, `XSutils.c`, `XSbench_header.h`, `make.bat`. 

We omit presenting the files for space. For your convenience, we have the files in `<https://github.com/ytopt-team/ytopt/tree/tutorial/ytopt/benchmark/xsbench-omp/xsbench>`. output lines are truncated 

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
```

Our search space contains three parameters: 1) `p0`: number of threads, 2) `p1`: block size for openmp dynamic schedule, 3) `p2`: turn on/off omp parallel.  


```python
# create an object of ConfigSpace 
cs = CS.ConfigurationSpace(seed=1234)
# number of threads
p0= CSH.OrdinalHyperparameter(name='p0', sequence=['4','5','6','7','8'], default_value='8')
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
Then, we need to define the objective function to evaluate a point in the search space. 

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
        cmd1 = "clang -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 " + \
        " -o " + tmpbinary + " " + interimfile +" " + kernel_dir + "/Materials.c " \
        + kernel_dir + "/XSutils.c " + " -I" + kernel_dir + \
        " -lm" + " -L${CONDA_PREFIX}/lib"
        cmd2 = kernel_dir + "/exe.pl " + tmpbinary
        
        #Find the compilation status using subprocess
        compilation_status = subprocess.run(cmd1, shell=True, stderr=subprocess.PIPE)

        #Find the execution time only when the compilation return code is zero, else return infinity
        if compilation_status.returncode == 0 :
            execution_status = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE)
            exetime = float(execution_status.stdout.decode('utf-8'))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime 
```

This file consists of several components.

`__init__()` take paths of the source file and output directory, and creates the output directory if it does not exists.   


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

`plotValues()` replace the Markers in the source file with the corresponding prameter values of the parameter dictionary.  


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

`findRuntime()` generates commandlines for compiling the source code and and executing the compiled code. Then, it finds the compilation status using subprocess; finds the execution time of the compiled code; and returns the execution time as cost to the search module. 


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
    cmd1 = "clang -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 " + \
    " -o " + tmpbinary + " " + interimfile +" " + kernel_dir + "/Materials.c " \
    + kernel_dir + "/XSutils.c " + " -I" + kernel_dir + \
    " -lm" + " -L${CONDA_PREFIX}/lib"
    cmd2 = kernel_dir + "/exe.pl " + tmpbinary

    #Find the compilation status using subprocess
    compilation_status = subprocess.run(cmd1, shell=True, stderr=subprocess.PIPE)

    #Find the execution time only when the compilation return code is zero, else return infinity
    if compilation_status.returncode == 0 :
        execution_status = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE)
        exetime = float(execution_status.stdout.decode('utf-8'))
        if exetime == 0:
            exetime = 1
    else:
        print(compilation_status.stderr)
        print("compile failed")
    return exetime #return execution time as cost
```

Note: you need to define your own evaluating function such as above. 


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

`python -m ytopt.search.ambs --evaluator ray --problem ytopt.benchmark.xsbench-omp.xsbench.problem.Problem --max-evals=10 --learner RF
`

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

Further Reading
--------------
- `xxx <yyy>`
