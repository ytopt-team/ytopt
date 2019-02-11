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
docs/	
    Sphinx documentation files
ppo/
    proximal policy optimization based reinforcement learning 
problems/
    easy to evalaute benchmark functions
test/
    scipts for running benchmark problems in the problems directory
ytopt/	
    scripts that contain the search implementations  
```

# Install instructions

```
conda create -n ytopt -c anaconda python=3.6
source activate ytopt
git clone https://github.com/ytopt-team/ytopt.git
cd ytopt/
pip install -e .
```

If you encounter mpi4py installtion error, (re)install mpich as follows
```
conda install -c conda-forge mpich
```
# Autotuning problem definition

An example is given in problems/ackley_mix

To define an autotuning problem, create two files.

The problem.py file defines the search space:

```
from collections import OrderedDict
import numpy as np
import os 

np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))

from ytopt.problem import Problem

cmd_frmt = "python " +HERE+"/executable.py"
nparam = 6

for i in range(0, nparam):
    cmd_frmt += f" --p{i} {'{}'}"
problem = Problem(cmd_frmt)

a, b = -15, 30

problem.spec_dim(p_id=0, p_space=(a, b), default=a)
problem.spec_dim(p_id=1, p_space=(a, b), default=a)
problem.spec_dim(p_id=2, p_space=[a+i for i in range(b-a)], default=a)
problem.spec_dim(p_id=3, p_space=[a+i for i in range(b-a)], default=a)
problem.spec_dim(p_id=4, p_space= list(np.random.permutation([str(a+i) for i in range(b-a)])), default=str(a))
problem.spec_dim(p_id=5, p_space= list(np.random.permutation([str(a+i) for i in range(b-a)])), default=str(a))

problem.checkcfg()

if __name__ == '__main__':
    print(problem)

```

The executable.py file defines the method to evaluate a point in the search space:
```
#!/usr/bin/env python
from __future__ import print_function
import re
import os
import sys
import time
import json
import math
import os
import argparse
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
seed = 12345

def create_parser():
    'command line parser'
    
    parser = argparse.ArgumentParser(add_help=True)
    group = parser.add_argument_group('required arguments')
    parser.add_argument('--p0', action='store', dest='p0',
                        nargs='?', const=2, type=float, default=-15.0,
                        help='parameter p0 value')
    parser.add_argument('--p1', action='store', dest='p1',
                        nargs='?', const=2, type=float, default=-15.0,
                        help='parameter p1 value')
    parser.add_argument('--p2', action='store', dest='p2',
                        nargs='?', const=2, type=int, default=-15,
                        help='parameter p2 value')
    parser.add_argument('--p3', action='store', dest='p3',
                        nargs='?', const=2, type=int, default=-15,
                        help='parameter p3 value')
    parser.add_argument('--p4', action='store', dest='p4',
                        nargs='?', const=2, type=str, default='-15',
                        help='parameter p4 value')
    parser.add_argument('--p5', action='store', dest='p5',
                        nargs='?', const=2, type=str, default='-15',
                        help='parameter p5 value')

    return(parser)

parser = create_parser()
cmdline_args = parser.parse_args()
param_dict = vars(cmdline_args)
print(param_dict)
p0 = param_dict['p0']
p1 = param_dict['p1']
p2 = param_dict['p2']
p3 = param_dict['p3']
p4 = int(param_dict['p4'])
p5 = int(param_dict['p5'])


x=np.array([p0,p1,p2,p3,p4,p5])

def ackley( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum(cos( c * x ))
    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)

pval = ackley(x, a=20, b=0.2, c=2*pi)
print('OUTPUT:%1.3f'%pval)
```


# Running

Reinforcement learning based search with proximal policy optimization
```
mpirun -np 2 python -m ytopt.search.ppo_a3c --prob_path=<PROBLEM_DIR_PATH>/problem.py --exp_dir=<EXP_DIR_PATH> --prob_attr=problem --exp_id=<ID>  --max_time=60 --base_estimator='PPO' 
```

Bayesian optimization with random forest model
```
mpirun -np 2 python -m ytopt.search.async_search --prob_path=<PROBLEM_DIR_PATH>/problem.py --exp_dir=<EXP_DIR_PATH> --prob_attr=problem --exp_id=<ID>  --max_time=60 --base_estimator='RF' 
```

Random search
```
mpirun -np 2 python -m ytopt.search.async_search --prob_path=<PROBLEM_DIR_PATH>/problem.py --exp_dir=<EXP_DIR_PATH> --prob_attr=problem --exp_id=<ID> --max_time=60 --base_estimator='DUMMY'
```

# How do I learn more?

<!-- * Documentation: https://ytopt.readthedocs.io -->

* GitHub repository: https://github.com/ytopt-team/ytopt


# Who is responsible?

The core ``ytopt`` team is at Argonne National Laboratory:

* Prasanna Balaprakash <pbalapra@anl.gov>, Lead and founder
* Romain Egele <regele@anl.gov>
* Paul Hovland <hovland@anl.gov>

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

# Copyright and license

TBD
