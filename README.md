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
test/
    scipts for running benchmark problems in the problems directory
ytopt/	
    scripts that contain the search implementations  
ytopt/benchmark/	
    a set of problems the user can use to compare our different search algorithms or as examples to build their own problems
```

# Install instructions
The autotuning framework requires the following components: Ytopt, scikit-optimize, and autotune. 

* We recommend creating isolated Python environments on your local machine using [conda](https://docs.conda.io/projects/conda/en/latest/index.html), for example:

```
conda create --name ytune python=3.7
conda activate ytune
```

* Install [scikit-optimize](https://github.com/pbalapra/scikit-optimize.git):
```
git clone https://github.com/pbalapra/scikit-optimize.git
cd scikit-optimize
pip install -e .
```

* Install [autotune](https://github.com/ytopt-team/autotune.git):
```
git clone -b version1 https://github.com/ytopt-team/autotune.git
cd autotune/
pip install -e . 
```

* Install [ytopt](https://github.com/ytopt-team/ytopt.git):
```
git clone https://github.com/ytopt-team/ytopt.git
cd ytopt/
pip install -e .
```

If you encounter installtion error, install psutil, setproctitle, mpich, mpi4py first as follows:
```
conda install -c conda-forge psutil
conda install -c conda-forge setproctitle
conda install -c conda-forge mpich
conda install -c conda-forge mpi4py
pip install -e .
```
# Defining autotuning problem

An example to autotune the OpenMP version of XSBench:

* You can define your search problem such as [ytopt/benchmark/xsbench-omp/xsbench/problem.py](https://github.com/jke513/ytopt/blob/master/ytopt/benchmark/xsbench-omp/xsbench/problem.py) for the following search space:

```
# number of threads
p0= CSH.OrdinalHyperparameter(name='p0', sequence=['4','5','6','7','8'], default_value='8')
#block size for openmp dynamic schedule
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['10','20','40','64','80','100','128','160','200'], default_value='100')
#clang unrolling
#omp parallel
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma omp parallel for", " "], default_value=' ')
cs.add_hyperparameters([p0, p1, p2])
```

* You can define the method to evaluate a point in the search space such as [ytopt/benchmark/xsbench-omp/plopper/plopper.py](https://github.com/jke513/ytopt/blob/master/ytopt/benchmark/xsbench-omp/plopper/plopper.py) for code generation and compiling.

* Note that you can install openmpi openmpi-mpicc openmp for this example:
```
conda install -c conda-forge openmp openmpi openmpi-mpicc
```

An example to autotune the OpenMP version of XSBench is given in [ytopt/benchmark/xsbench-mpi-omp/xsbench/](https://github.com/jke513/ytopt/blob/master/ytopt/benchmark/xsbench-mpi-omp/xsbench/).

<!-- An example to autotune the deep learning mnist problem is given in [ytopt/benchmark/dl/](https://github.com/jke513/ytopt/tree/master/ytopt/benchmark/dl).

 You can define your search problem such as:

* An example to autotune the OpenMP version of XSBench is given in [ytopt/benchmark/xsbench-omp/xsbench/problem.py](https://github.com/jke513/ytopt/blob/master/ytopt/benchmark/xsbench-omp/xsbench/problem.py).

```
cs = CS.ConfigurationSpace(seed=1234)
# number of threads
p0= CSH.OrdinalHyperparameter(name='p0', sequence=['4','5','6','7','8'], default_value='8')
#block size for openmp dynamic schedule
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['10','20','40','64','80','100','128','160','200'], default_value='100')
#clang unrolling
#omp parallel
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma omp parallel for", " "], default_value=' ')

cs.add_hyperparameters([p0, p1, p2])
```



* An example to autotune the hybrid MPI/OpenMP version of XSBench is given in [ytopt/benchmark/xsbench-mpi-omp/xsbench/problem.py](https://github.com/jke513/ytopt/blob/master/ytopt/benchmark/xsbench-mpi-omp/xsbench/problem.py).

```

``` -->


# Running

Bayesian optimization with random forest model:
```
python -m ytopt.search.ambs --evaluator ray --problem ytopt.benchmark.xsbench-omp.xsbench.problem.Problem --max-evals=10 --learner RF
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
