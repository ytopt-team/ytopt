<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

<!-- [![Documentation Status](https://readthedocs.org/projects/ytopt/badge/?version=latest)](https://ytopt.readthedocs.io/en/latest/?badge=latest)-->

# What is ytopt?
``ytopt`` is a machine-learning-based search software package that consists of sampling a small number of input parameter configurations, evaluating them, and progressively fitting a surrogate model over the input-output space until exhausting the user-defined time or the maximum number of evaluations. The package is built based on Bayesian Optimization that solves any optimization problem and is especially useful when the objective function is hard to evaluate. It provides an interface that deals with unconstrained and constrained optimization problems. The software is designed to operate in the master-worker computational paradigm, where one master node fits the surrogate model and generates promising input configurations and worker nodes perform the computationally expensive evaluations and return the outputs to the master node. The asynchronous aspect of the search allows the search to avoid waiting for all the evaluation results before proceeding to the next iteration. As soon as an evaluation is finished, the data is used to retrain the surrogate model, which is then used to bias the search toward the promising configurations. 

<!--
``ytopt`` is a machine-learning-based search software package that consists of sampling a small number of input parameter configurations,
evaluating them, and progressively fitting a surrogate model over the input-output space until exhausting the user-defined time or maximum number of 
evaluations. The package provides two different class of methods: Bayesian Optimization and Reinforcement Learning.
The software is designed to operate in the master-worker computational paradigm, where one master node fits 
the surrogate model and generates promising input configurations and worker nodes perform the computationally expensive evaluations and 
return the outputs to the master node.
The asynchronous aspect of the search allows the search to avoid waiting for all the evaluation results before proceeding to the next iteration. As 
soon as an evaluation is finished, the data is used to retrain the surrogate model, which is then used to bias the search toward the promising configurations. -->
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
The autotuning framework requires the following components: ConfigSpace, scikit-optimize, autotune, and ytopt. 

* We recommend creating isolated Python environments on your local machine using [conda](https://docs.conda.io/projects/conda/en/latest/index.html), for example:

```
conda create --name ytune python=3.7
conda activate ytune
```

* Create a directory for ytopt tutorial as follows:
```
mkdir ytopt
cd ytopt
```

* Install [ConfigSpace](https://github.com/deephyper/ConfigSpace.git):
```
git clone https://github.com/deephyper/ConfigSpace.git configspace
cd configspace
pip install -e .
cd ..
```

* Install [scikit-optimize](https://github.com/deephyper/scikit-optimize.git):
```
git clone https://github.com/deephyper/scikit-optimize.git
cd scikit-optimize
pip install -e .
cd ..
```

* Install [autotune](https://github.com/ytopt-team/autotune.git):
```
git clone -b version1 https://github.com/ytopt-team/autotune.git
cd autotune
pip install -e . 
cd ..
```

* Install [ytopt](https://github.com/ytopt-team/ytopt.git):
```
git clone -b main https://github.com/ytopt-team/ytopt.git
cd ytopt
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

# Tutorials

* [Autotuning the block matrix multiplication](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mmm-block/tutorial-mmm-block.md)
* [Autotuning the OpenMP version of XSBench](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.md)
* [Autotuning the OpenMP version of XSBench with constraints](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/tutorial-omp-xsbench-const.md)
* [Autotuning the hybrid MPI/OpenMP version of XSBench](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mpi-omp-xsbench/tutorial-mpi-omp-xsbench.md)
* [Autotuning the hybrid MPI/OpenMP version of XSBench with constraints](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mpi-omp-xsbench/tutorial-mpi-omp-xsbench-const.md)
* [Autotuning the OpenMP version of convolution-2d with constraints](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/convolution-2d/tutorial-convolution-2d-const.md)

<!--# How do I learn more?

* Documentation: https://ytopt.readthedocs.io 

* GitHub repository: https://github.com/ytopt-team/ytopt -->


# Who is responsible?

The core ``ytopt`` team is at Argonne National Laboratory:

* Prasanna Balaprakash <pbalapra@anl.gov>, Lead and founder
* Romain Egele <regele@anl.gov>
* Paul Hovland <hovland@anl.gov>
* Xingfu Wu <xingfu.wu@anl.gov>
* Jaehoon Koo <jkoo@anl.gov>

<!--Modules, patches (code, documentation, etc.) contributed by:

* David Fridlander <davidfrid2@gmail.com>

# How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to:

* Our mailing list: *ytopt@groups.io* or https://groups.io/g/ytopt

* Issues on GitHub

Patches are much appreciated on the software itself as well as documentation.
Optionally, please include in your first patch a credit for yourself in the
list above.

The ytopt team uses git-flow to organize the development: [Git-Flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/). For tests we are using: [Pytest](https://docs.pytest.org/en/latest/).-->

# Acknowledgements

* YTune: Autotuning Compiler Technology for Cross-Architecture Transformation and Code Generation, U.S. Department of Energy Exascale Computing Project (2017--Present) 
* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--Present)
* PROTEAS-TUNE, U.S. Department of Energy ASCR Exascale Computing Project (2018--Present)

# Copyright and license

TBD
