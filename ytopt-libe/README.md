# Autotuning Framework ytopt-libe
[ytopt](https://github.com/ytopt-team/ytopt.git) is a machine-learning-based search software package that consists of sampling a small number of input parameter configurations, evaluating them, and progressively fitting a surrogate model over the input-output space until exhausting the user-defined time or the maximum number of evaluations. The package is built based on Bayesian Optimization that solves any optimization problem and is especially useful when the objective function is difficult to evaluate. It provides an interface that deals with unconstrained and constrained problems. The software is designed to operate in the manager-worker computational paradigm, where one manager node fits the surrogate model and generates promising input configurations and worker nodes perform the computationally expensive evaluations and return the outputs to the manager node. The asynchronous aspect of the search allows the search to avoid waiting for all the evaluation results before proceeding to the next iteration. As soon as an evaluation is finished, the data is used to retrain the surrogate model, which is then used to bias the search toward the promising configurations.

[libensemble](https://github.com/ytopt-team/libensemble.git) is a Python toolkit for coordinating workflows of asynchronous and dynamic ensembles of calculations in parallel. It helps users take advantage of massively parallel resources to solve design, decision, and inference problems and expands the class of problems that can benefit from increased parallelism. libensemble employs a manager/worker scheme that communicates via MPI, multiprocessing, or TCP. A manager allocate work to multiple workers. Workers control and monitor any level of work from small subnode tasks to huge many-node computations.

By integrating libensemble with ytopt, the autotuning framework ytopt-libe not only accelerates the evaluation process of ytopt in parallel but also improves the accuracy of Random Forests surrogate model by feeding more data to make the search more efficient.

# Directory

This directory includes the ECP apps, ECP proxy apps, and deep learning applications in pytorch/tensorflow autotuned using ytopt-libe and ytopt only.

```
deeplearning/
    Autotuning deep learning applications 
xsbench/
    Autotuning ECP proxy app XSBench
sw4lite/
    Autotuning ECP proxy app SW4lite
tvm/
    Autotuning Apache TVM (Tensor Virtual Machine)-based scientific applications (3mm, lu, cholesky)
svms/
    Autotuning SVM (Support Vector Machine)-based scientific simulations
```

# Installation instructions
The autotuning framework ytopt-libe requires the following components: ConfigSpace,scikit-optimize, autotune, ytopt, and libensemble.

* We recommend creating isolated Python environments (python >=3.7) on your local machine using an up to date [conda](https://docs.conda.io/projects/conda/en/latest/index.html), for example:

```
conda create --name ytune python=3.10
conda activate ytune
```

* Create a directory for installing all required packages for ytopt-libe as follows:
```
mkdir ytune
cd ytune
```

* Install [ConfigSpace](https://github.com/ytopt-team/ConfigSpace.git):
```
pip install configspace==0.7.1
```

* Install [scikit-optimize](https://github.com/ytopt-team/scikit-optimize.git):
```
git clone https://github.com/ytopt-team/scikit-optimize.git
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
git clone https://github.com/ytopt-team/ytopt.git
cd ytopt
pip install -e .
cd ..
```

* Install [libensemble](https://github.com/ytopt-team/libensemble.git):
```
git clone https://github.com/ytopt-team/libensemble.git
cd libensemble
pip install -e .
cd ..
```

* Install examples of [ytopt-libe](https://github.com/ytopt-team/ytopt-libensemble.git):
```
git clone https://github.com/ytopt-team/ytopt-libensemble.git
cd ytopt-libensemble
```

After installing ConfigSpace, Scikit-optimize, autotune, ytopt, and libensemble successfully, the autotuning framework ytopt-libe is ready to use.
If you have some issues about ConfigSpace, just downgrade configspace by the following command line "pip install configspace==0.7.1" in your conda environment to solve these issues.

* Example: Using ytopt-libe to autotune the  MPI/OpenMP version of XSBench:
```
cd ytopt-libe-xsbench
cd laptop
* If you want to change the compiler mpicc (default), edit the file plopper.py. 
* Make sure to create the conda environemnt ytune before running a test
* Modify the run script runs.sh with the proper conda environment, number of wokers, MPI ranks, and the application timeout (recommanded to use the script to make these settings)
* Then, use the run script to autotune XSBench 

./runs.sh
```

Note: For the diagnosis purpose, look at the log files (*.log) or text files (*.txt) for any error under the current directory.

# Tutorials

* [Autotuning the MPI/OpenMP version of XSBench](https://github.com/ytopt-team/ytopt-libensemble/tree/main/ytopt-libe-xsbench)
* [Autotuning the GPU version of OpenMC](https://github.com/ytopt-team/ytopt-libensemble/tree/main/ytopt-libe-openmc)
* [Autotuning Deep Learning Applications](https://github.com/ytopt-team/ytopt-libensemble/tree/develop1/ytopt-libe-deeplearning)

# Publications
* X. Wu, P. Balaprakash, M. Kruse, J. Koo, B. Videau, P. Hovland, V. Taylor, B. Geltz, S. Jana, and M. Hall, "ytopt: Autotuning Scientific Applications for Energy Efficiency at Large Scales", Concurrency and Computation: Practice and Experience, Oct. 2024. DOI: [10.1002/cpe.8322](https://onlinelibrary.wiley.com/doi/10.1002/cpe.8322).
 
* X. Wu, J. R. Tramm, J. Larson, J.-L. Navarro, P. Balaprakash, B. Videau, M. Kruse, P. Hovland, V. Taylor, and M. Hall, "Integrating ytopt and libEnsemble to Autotune OpenMC", DOI: [10.48550/arXiv.2402.09222](https://doi.org/10.48550/arXiv.2402.09222),  International Journal of High Performance Computing Applications, DOI: [10.1177/10943420241286476](https://journals.sagepub.com/doi/10.1177/10943420241286476), Oct. 2024.
* X. Wu and T. Oli and J. H. Qian and V. Taylor and M. C. Hersam and V. K. Sangwan, "An Autotuning-based Optimization Framework for Mixed-kernel SVM Classifications in Smart Pixel Datasets and Heterojunction Transistors", DOI: [10.48550/arXiv.2406.18445](https://arxiv.org/pdf/2406.18445), 2024.
* X. Wu, P. Paramasivam, and V. Taylor, "Autotuning Apache TVM-based Scientific Applications Using Bayesian Optimization", SC23 Workshop on Artificial Intelligence and Machine Learning for Scientific Applications (AI4S’23), Nov. 13, 2023, Denver, CO. https://arxiv.org/pdf/2309.07235.pdf.

# Acknowledgements
* SciDAC RAPIDS and OASIS, U.S. Department of Energy ASCR (2024--present)
* PROTEAS-TUNE, U.S. Department of Energy ASCR Exascale Computing Project (2018--2023)

