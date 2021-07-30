## Autotuning tutorials

A tutorial to autotune the block matrix multiplication is given in [/docs/tutorials/mmm-block](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mmm-block/)

* You can follow the [/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.ipynb](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.ipynb) or the [/docs/tutorials/mmm-block/tutorial-mmm-block.md](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mmm-block/tutorial-mmm-block.md) for this tutorial. 

* You can define your search problem such as [ytopt/benchmark/mmm-block/mmm_problem/problem.py](https://github.com/ytopt-team/ytopt/tree/ytopt/benchmark/mmm-block/mmm_problem/problem.py) for the following search space:

* You can define the method to evaluate a point in the search space such as [ytopt/benchmark/mmm-block/plopper/plopper.py](https://github.com/ytopt-team/ytopt/tree/ytopt/benchmark/mmm-block/plopper/plopper.py) for code generation and compiling.

* Bayesian optimization with random forest model:
```
python -m ytopt.search.ambs --evaluator ray --problem ytopt.benchmark.mmm-block.mmm_problem.problem.Problem --max-evals=5 --learner RF
```
* Then, ytopt.log, results.csv, and results.json will be rendered. 

* This example including the source code is borrowed from [http://opentuner.org/tutorial/gettingstarted/](http://opentuner.org/tutorial/gettingstarted/). 

A tutorial to autotune the OpenMP version of XSBench is given in [/docs/tutorials/omp-xsbench](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/)

* You can follow the [/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.ipynb](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.ipynb) or the [/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.md](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.md) for this tutorial. 

* You can define your search problem such as [ytopt/benchmark/xsbench-omp/xsbench/problem.py](https://github.com/ytopt-team/ytopt/tree/ytopt/benchmark/xsbench-omp/xsbench/problem.py) for the following search space:

* You can define the method to evaluate a point in the search space such as [ytopt/benchmark/xsbench-omp/plopper/plopper.py](https://github.com/ytopt-team/ytopt/tree/ytopt/benchmark/xsbench-omp/plopper/plopper.py) for code generation and compiling.

* Note that you can install openmpi openmpi-mpicc openmp for this example:
```
conda install -c conda-forge openmp openmpi openmpi-mpicc
```

* Bayesian optimization with random forest model:
```
python -m ytopt.search.ambs --evaluator ray --problem ytopt.benchmark.xsbench-omp.xsbench.problem.Problem --max-evals=10 --learner RF
```
* Then, ytopt.log, results.csv, and results.json will be rendered. 

<!--
An example to autotune the hybrid MPI/OpenMP version of XSBench is given in [ytopt/benchmark/xsbench-mpi-omp/xsbench/](https://github.com/jke513/ytopt/blob/master/ytopt/benchmark/xsbench-mpi-omp/xsbench/).

 An example to autotune the deep learning mnist problem is given in [ytopt/benchmark/dl/](https://github.com/jke513/ytopt/tree/master/ytopt/benchmark/dl).

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


