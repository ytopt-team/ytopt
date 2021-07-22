<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

[![Documentation Status](https://readthedocs.org/projects/ytopt/badge/?version=latest)](https://ytopt.readthedocs.io/en/latest/?badge=latest)

# What is ytopt?
``ytopt`` is a machine-learning-based search software package that consists of sampling a small number of input parameter configurations,
evaluating them, and progressively fitting a surrogate model over the input-output space until exhausting the user-defined time or maximum number of 
evaluations. The package provides two different class of methods: Bayesian Optimization and Reinforcement Learning.
The software is designed to operate in the master-worker computational paradigm, where one master node fits 
the surrogate model and generates promising input configurations and worker nodes perform the computationally expensive evaluations and 
return the outputs to the master node.
The asynchronous aspect of the search allows the search to avoid waiting for all the evaluation results before proceeding to the next iteration. As 
soon as an evaluation is finished, the data is used to retrain the surrogate model, which is then used to bias the search toward the 
promising configurations. 

# Directory structure
```
Benchmarks/	
    a set of problems the user can use to compare our different search algorithms or as examples to build their own problems
docs/	
    Sphinx documentation files
test/
    scipts for running benchmark problems in the problems directory
ytopt/	
    scripts that contain the search implementations  
```

# Install instructions
The autotuning framework requires the following components: Ytopt, Configspace, scikit-optimize, and autotune. 

We recommend creating isolated Python environments on your local machine using virtualenv, for example:

```
conda create --name ytune python=3.7
conda activate ytune
```

Install Configspace:
```
pip install ConfigSpace 
```

Install scikit-optimize:
```
git clone https://github.com/pbalapra/scikit-optimize.git
cd scikit-optimize
pip install -e .
```

Install autotune:
```
git clone -b version1  https://github.com/ytopt-team/autotune.git
cd autotune/
pip install -e . 
```

Install ytopt:
```
git clone https://github.com/ytopt-team/ytopt.git
cd ytopt/
pip install -e .
pip install scikit-learn==0.23.1
```

If you encounter installtion error, install psutil, setproctitle, mpich, mpi4py first as follows:
```
conda install -c conda-forge psutil
conda install -c conda-forge setproctitle
conda install -c conda-forge mpich
conda install -c conda-forge mpi4py
pip install -e .
```
# Autotuning problem definition

1. An example for hyperparameter search of the nerual network on mnist is given in /Benchmark/DL/mnist/problem.py

The problem.py file defines the search space:
```
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *
import os
import sys
import time
import json
import math

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper
nparams = 4

cs = CS.ConfigurationSpace(seed=1234)
#batch_size
p0= CSH.OrdinalHyperparameter(name='p0', sequence=['16','32','64','100','128','200','256','300','400','512'], default_value='128')
#epochs
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['1','2','4','8','12','16','20','22','24','30'], default_value='20')
#dropout rate
p2= CSH.OrdinalHyperparameter(name='p2', sequence=['0.1', '0.15', '0.2', '0.25','0.4'], default_value='0.2')
#optimizer
p3= CSH.CategoricalHyperparameter(name='p3', choices=['rmsprop','adam','sgd','adamax','adadelta','adagrad','nadam'], default_value='rmsprop')

cs.add_hyperparameters([p0, p1, p2, p3])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/dlp.py',dir_path)

x1=['p0','p1','p2','p3']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]]]
    print('VALUES:',point[x1[0]])
    params = ["P1","P2","P3","P4"]

    result = obj.findRuntime(value, params)
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT: ',results)

  return results

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
```

2. An example for loop optimization problem with constraints is given in ytopt/benchmark/loopopt/problem.py

The problem.py file defines the search space:
```
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *

import sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical

cs = CS.ConfigurationSpace(seed=1234)
p1 = CSH.CategoricalHyperparameter(name='p1', choices=['None', '#pragma omp #p3', '#pragma omp target #p3', '#pragma omp target #p5', '#pragma omp #p4'])
p3 = CSH.CategoricalHyperparameter(name='p3', choices=['None', '#parallel for #p4', '#parallel for #p6', '#parallel for #p7'])
p4 = CSH.CategoricalHyperparameter(name='p4', choices=['None', 'simd'])
p5 = CSH.CategoricalHyperparameter(name='p5', choices=['None', '#dist_schedule static', '#dist_schedule #p11'])
p6 = CSH.CategoricalHyperparameter(name='p6', choices=['None', '#schedule #p10', '#schedule #p11'])
p7 = CSH.CategoricalHyperparameter(name='p7', choices=['None', '#numthreads #p12'])
p10 = CSH.CategoricalHyperparameter(name='p10', choices=['static', 'dynamic'])
p11 = CSH.OrdinalHyperparameter(name='p11', sequence=['1', '8', '16'])
p12 = CSH.OrdinalHyperparameter(name='p12', sequence=['1', '8', '16'])

cs.add_hyperparameters([p1, p3, p4, p5, p6, p7, p10, p11, p12])

#make p3 an active parameter when p1 value is ... 
cond0 = CS.EqualsCondition(p3, p1, '#pragma omp #p3')
cond1 = CS.EqualsCondition(p3, p1, '#pragma omp target #p3')
cond2 = CS.EqualsCondition(p5, p1, '#pragma omp target #p5')
cond3 = CS.EqualsCondition(p4, p1, '#pragma omp #p4')
cond4 = CS.EqualsCondition(p4, p3, '#parallel for #p4')
cond5 = CS.EqualsCondition(p6, p3, '#parallel for #p6')
cond6 = CS.EqualsCondition(p7, p3, '#parallel for #p7')
cond7 = CS.EqualsCondition(p11, p5, '#dist_schedule #p11')
cond8 = CS.EqualsCondition(p10, p6, '#schedule #p10')
cond9 = CS.EqualsCondition(p11, p6, '#schedule #p11')
cond10 = CS.EqualsCondition(p12, p7, '#numthreads #p12')

cs.add_condition(CS.OrConjunction(cond0,cond1))
cs.add_condition(cond2) 
cs.add_condition(CS.OrConjunction(cond3,cond4))
cs.add_condition(cond5) 
cs.add_condition(cond6) 
cs.add_condition(cond8) 
cs.add_condition(CS.OrConjunction(cond7,cond9))
cs.add_condition(cond10)

# problem space
task_space = None
input_space = cs 

output_space = Space([
    Real(-inf, inf, name='y')
])

def myobj(point: dict):
    s = np.random.uniform()
    return s

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
``` 


# Running

Bayesian optimization with random forest model:
```
python -m ytopt.search.ambs --evaluator ray --problem ytopt.benchmark.loopopt.problem.Problem --learner RF
```

# How do I learn more?

<!-- * Documentation: https://ytopt.readthedocs.io -->

* GitHub repository: https://github.com/ytopt-team/ytopt


# Who is responsible?

The core ``ytopt`` team is at Argonne National Laboratory:

* Prasanna Balaprakash <pbalapra@anl.gov>, Lead and founder
* Romain Egele <regele@anl.gov>
* Paul Hovland <hovland@anl.gov>
* Xingfu Wu <xingfu.wu@anl.gov>
* Jaehoon Koo <jkoo@anl.gov>

Modules, patches (code, documentation, etc.) contributed by:

# How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to:

* Our mailing list: *ytopt@groups.io* or https://groups.io/g/ytopt

* Issues on GitHub

Patches are much appreciated on the software itself as well as documentation.
Optionally, please include in your first patch a credit for yourself in the
list above.

The ytopt team uses git-flow to organize the development: [Git-Flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/). For tests we are using: [Pytest](https://docs.pytest.org/en/latest/).

# Acknowledgements

* YTune: Autotuning Compiler Technology for Cross-Architecture Transformation and Code Generation, U.S. Department of Energy Exascale Computing Project (2017--Present) 
* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--Present)
* PROTEAS-TUNE, U.S. Department of Energy ASCR Exascale Computing Project (2018--Present)

# Copyright and license

TBD
