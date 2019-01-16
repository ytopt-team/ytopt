Directory structure
===================
```
benchmarks
    directory for problems
experiments
    directory for saving the running the experiments and storing the results
search
    directory for source files
```
Install instructions
====================

With anaconda do the following:

```
conda create --name ytopt -c intel intelpython3_core python=3.6
source activate ytopt
conda install h5py
conda install scikit-learn
conda install pandas
conda install mpi4py
conda install -c conda-forge keras
conda install -c conda-forge scikit-optimize
git clone https://github.com/scikit-optimize/scikit-optimize.git
cd scikit-optimize
pip install -e.
```
Usage
=====
```
cd search

usage: async-search.py [-h] [-v] [--prob_dir [PROB_DIR]] [--exp_dir [EXP_DIR]]
                       [--exp_id [EXP_ID]] [--max_evals [MAX_EVALS]]
                       [--max_time [MAX_TIME]]

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  --prob_dir [PROB_DIR]
                        problem directory
  --exp_dir [EXP_DIR]   experiments directory
  --exp_id [EXP_ID]     experiments id
  --max_evals [MAX_EVALS]
                        maximum number of evaluations
  --max_time [MAX_TIME]
                        maximum time in secs
```
Example
=======
```
mpiexec -np 2 python async-search.py --prob_dir=../benchmarks/prob  --exp_dir=../experiments/ --exp_id=exp-01 --max_evals=10 --max_time=60
```
How to define your own autotuning problem
=========================================
This will be illustrated with the example in /benchmarks/prob directory.

In this example, we want to tune the executable.py that gets four command line parameters and returns the output value
```
python executable.py --help
usage: executable.py [-h] [--p1 [P1]] [--p2 [P2]] [--p3 [P3]] [--p4 [P4]]

optional arguments:
  -h, --help  show this help message and exit
  --p1 [P1]   parameter p1 value
  --p2 [P2]   parameter p2 value
  --p3 [P3]   parameter p3 value
  --p4 [P4]   parameter p4 value
```
For example,
```
python executable.py --p1=2 --p2=2 --p3=4 --p4=a
OUTPUT:16.000
```

The search space and a default starting point is defined in problem.py

```
from collections import OrderedDict
class Problem():
    def __init__(self):
        space = OrderedDict()
        #problem specific parameters
        space['p1'] = (2, 10)
        space['p2'] = (8, 1024)
        space['p3'] = [2 , 4, 8, 16, 32, 64, 128]
        space['p4'] = ['a', 'b', 'c']
        self.space = space
        self.params = self.space.keys()
        self.starting_point = [2, 8, 2, 'c']
```
In evalaute.py, you have to define three functions.

First, define how to construct the command line in
```
def commandLine(x, params)
```

Second, define how to evalaute a point in
```
def evaluate(x, evalCounter, params, prob_dir, job_dir, result_dir):
```

Third, define how to read the results in
```
def readResults(fname, evalnum):
```

Finally, in job.tmpl, call the executable (see the example)