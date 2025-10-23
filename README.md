<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

<!-- [![Documentation Status](https://readthedocs.org/projects/ytopt/badge/?version=latest)](https://ytopt.readthedocs.io/en/latest/?badge=latest)-->

# What is ytopt?



``ytopt`` is a machine learning-based autotuning and hyperparameter optimization software package that uses Bayesian Optimization to find the best input parameter/hyperparameter configurations for a given kernel, miniapp, or application with the best system configurations for a given HPC system.

``ytopt`` accepts the following as input:

  1. A code-evaluation wrapper with tunable parameters as a code mold for performance measurement 
  2. Tunable application parameters (hyperparameters) and tunable system parameters 
  3. The corresponding parameter search space for the tunable parameters

By sampling and evaluating a small number of input configurations, ``ytopt`` gradually builds a surrogate model of the input-output space. This process continues until the user-specified time or the maximum number of evaluations is reached.

``ytopt`` handles both unconstrained and constrained optimization problems, searches asynchronously, and can look-ahead on iterations to more effectively adapt to new evaluations and adjust the search towards promising configurations, leading to a more efficient and faster convergence on the best solutions.

Internally, ``ytopt`` uses a manager-worker computational paradigm, where one node fits the surrogate model and generates new input configurations, and other nodes perform the computationally expensive evaluations and return the results to the manager node. This is implemented in two ways: using [ray](https://github.com/ray-project/ray) for ``ytopt/benchmark`` in sequential processing and  using [libensemble](https://github.com/Libensemble/libensemble) for ``ytopt-libe`` in parallel processing.

Additional documentation is available on [Read the Docs](https://ytopt.readthedocs.io/en/latest/). Access ``ytopt-libe`` for the latest examples with new features and development.

# Installation instructions
``ytopt`` requires the following components: ``dh-scikit-optimize``, ``autotune``, and ``ConfigSpace``. When ytopt is being installed, ``ConfigSpace`` and ``LibEnsemble`` are required to be installed automatically.


* We recommend creating isolated Python environments on your local machine using [conda](https://docs.conda.io/projects/conda/en/latest/index.html), for example:

```
conda create --name ytune python=3.10
conda activate ytune
```

* Create a directory for ``ytune``:
```
mkdir ytune
cd ytune
```

* Install [dh-scikit-optimize](https://github.com/ytopt-team/scikit-optimize.git):
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
git clone -b main https://github.com/ytopt-team/ytopt.git
cd ytopt
pip install -e .
```

After installing scikit-optimize, autotune, and ytopt successfully, the autotuning framework ytopt is ready to use. Browse the ``ytopt/benchmark`` directory for an extensive collection of old examples, or encourage to access ``ytopt-libe`` for the latest examples with new features.

```
# [Optional] Install [CConfigSpace](https://github.com/argonne-lcf/CCS.git):
    * Prerequisites: ``autotools`` and ``gsl``
        * Ubuntu
          ```
          sudo apt-get install autoconf automake libtool libgsl-dev
          ```

        * MacOS
          ```
          brew install autoconf automake libtool gsl
          ```
    * Build and Install the library and python bindings:
      the `configure` command can take an optional `--prefix=` parameter to specify a
      different install path than the default one (`/usr/local`). Depending on the
      chosen location you may need elevated previleges to run `make install`.
      ```
      git clone git@github.com:argonne-lcf/CCS.git
      cd CCS
      ./autogen.sh
      mkdir build
      cd build
      ../configure
      make
      make install
      cd ../bindings/python
      pip install parglare==0.12.0
      pip install -e .
      ```
    * Setup environment:
      in order for the python binding to find the CConfigSpace library, the path to
      the library install location (`/usr/local/lib` by default) must be appended
      to the `LD_LIBRARY_PATH` environment variable on Linux, while on MacOS the
      `DYLD_LIBRARY_PATH` environment variable serves the same purpose. Alternatively
      the `LIBCCONFIGSPACE_SO_` environment variable can be made to point to the installed
      `libcconfigspace.so` file on Linux or to the installed `libcconfigspace.dylib`
      on MacOS. 

* [Optional] Install Online tuning:
    * Online tuning with transfer learning interface is built on Synthetic Data Vault (SDV):
    * Install [SDV](https://github.com/sdv-dev/SDV.git):
      ```
      cd ytopt
      pip install -e .[online]
      ```
    * For macOS it may need to do: ``pip install -e ".[online]"``  

```

# Directory structure
```
docs/
    Sphinx documentation files
test/
    scipts for running benchmark problems in the problems directory
ytopt/
    scripts that contain the search implementations
ytopt/hpo/
    Hyperparameter optimization with 7 and 17 hyperparameters using ray
ytopt/benchmark/
    a set of problems the user can use to compare our different search algorithms or as examples to build their own problems
ytopt/Benchmarks/
    a set of problems for autotuning PolyBench 4.2 and ECP proxy apps
ytopt-libe/
    scripts and a set of examples for using ytopt-libe with new features 
ytopt-libe/hpo/
    Hyperparameter optimization with 7 and 17 hyperparameters using libensemble
ytopt-libe/hpo4llm/
    Hyperparameter optimization for a loss function for LLM training
```

# Basic Usage

``ytopt`` is typically run from the command-line in the following example manner:

``python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=10 --learner RF``

Where:
  * The *search* variant is one of ``ambs`` (*Asynchronous Model-Based Search*) or ``async_search`` (run as an MPI process).
  * The *evaluator* is the method of concurrent evaluations, and can be ``ray`` or ``subprocess``.
  * The *problem* is typically an ``autotune.TuningProblem`` instance. Specify the module path and instance name.
  * ``--max-evals`` is the maximum number of evaluations.

Depending on the *search* variant chosen, other command-line options may be provided. For example, the ``ytopt.search.ambs`` search
method above was further customized by specifying the ``RF`` learning strategy.

See the [``autotune`` docs](https://github.com/ytopt-team/autotune) for basic information on getting started with creating a ``TuningProblem`` instance.

See the [``ConfigSpace`` docs](https://automl.github.io/ConfigSpace/main/) for guidance on defining input/output parameter spaces for problems.

Otherwise, access the subdirectory ``ytopt-libe`` or [ytopt-libensemble](https://github.com/ytopt-team/ytopt-libensemble) for the latest examples with new features.

# Tutorials

* [Autotuning the block matrix multiplication](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mmm-block/tutorial-mmm-block.md)
* [Autotuning the OpenMP version of XSBench](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/tutorial-omp-xsbench.md)
* [Autotuning the OpenMP version of XSBench with constraints](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/omp-xsbench/tutorial-omp-xsbench-const.md)
* [Autotuning the hybrid MPI/OpenMP version of XSBench](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mpi-omp-xsbench/tutorial-mpi-omp-xsbench.md)
* [Autotuning the hybrid MPI/OpenMP version of XSBench with constraints](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/mpi-omp-xsbench/tutorial-mpi-omp-xsbench-const.md)
* [Autotuning the OpenMP version of convolution-2d with constraints](https://github.com/ytopt-team/ytopt/tree/tutorial/docs/tutorials/convolution-2d/tutorial-convolution-2d-const.md)
* [(Optinal) Autotuning the OpenMP version of XSBench online](https://github.com/ytopt-team/ytopt/blob/online/docs/tutorials/omp-xsbench-tl/tutorial-omp-xsbench-tl.md)

<!--# How do I learn more?

* Documentation: https://ytopt.readthedocs.io 

* GitHub repository: https://github.com/ytopt-team/ytopt -->


# Who is responsible?

The core ``ytopt`` team is at Argonne National Laboratory:

* Xingfu Wu <xingfu.wu@anl.gov>
* Prasanna Balaprakash <pbalapra@anl.gov>
* Brice Videau <bvideau@anl.gov>
* Paul Hovland <hovland@anl.gov>
* Romain Egele <regele@anl.gov>
* Jaehoon Koo <jkoo@anl.gov>

The convolution-2d tutorial (source and python scripts) is contributed by:
* David Fridlander <davidfrid2@gmail.com>

<!--Modules, patches (code, documentation, etc.) contributed by:

* David Fridlander <davidfrid2@gmail.com>

# How can I participate?

Questions, comments, feature requests, bug reports, etc. can be directed to:

* Our mailing list: *ytopt@groups.io* or https://groups.io/g/ytopt

* Issues on GitHub

Patches are much appreciated on the software itself as well as documentation.
Optionally, please include in your first patch a credit for yourself in the
list above.

The ytopt team uses git-flow to organize the development: [Git-Flow cheatsheet](https://danielkummer.github.io/git-flow-cheatsheet/). For tests we are using: [Pytest](https://docs.pytest.org/en/latest/). -->

# Publications
* X. Wu, P. Balaprakash, M. Kruse, J. Koo, B. Videau, P. Hovland, V. Taylor, B. Geltz, S. Jana, and M. Hall, "ytopt: Autotuning Scientific Applications for Energy Efficiency at Large Scales", Concurrency and Computation: Practice and Experience, Vol. 37 (1): e8322, Jan. 2025.  DOI: [10.1002/cpe.8322](https://onlinelibrary.wiley.com/doi/10.1002/cpe.8322).
* X. Wu, J. R. Tramm, J. Larson, J.-L. Navarro, P. Balaprakash, B. Videau, M. Kruse, P. Hovland, V. Taylor, and M. Hall, "Integrating ytopt and libEnsemble to Autotune OpenMC", DOI: [10.48550/arXiv.2402.09222](https://doi.org/10.48550/arXiv.2402.09222),  International Journal of High Performance Computing Applications, Vol. 39, No. 1, 79-103, Jan. 2025. DOI: [10.1177/10943420241286476](https://journals.sagepub.com/doi/10.1177/10943420241286476).
* X. Wu and T. Oli and J. H. Qian and V. Taylor and M. C. Hersam and V. K. Sangwan, "An Autotuning-based Optimization Framework for Mixed-kernel SVM Classifications in Smart Pixel Datasets and Heterojunction Transistors", DOI: [10.48550/arXiv.2406.18445](https://arxiv.org/pdf/2406.18445), 2024.
* X. Wu, P. Paramasivam, and V. Taylor, "Autotuning Apache TVM-based Scientific Applications Using Bayesian Optimization", SC23 Workshop on Artificial Intelligence and Machine Learning for Scientific Applications (AI4S’23), Nov. 13, 2023, Denver, CO. https://arxiv.org/pdf/2309.07235.pdf.
* T. Randall, J. Koo, B. Videau, M. Kruse, X. Wu, P. Hovland, M. Hall, R. Ge, and P. Balaprakash. "Transfer-Learning-Based Autotuning Using Gaussian Copula". In 2023 International Conference on Supercomputing (ICS ’23), June 21–23, 2023, Orlando, FL, USA. ACM, New York, NY, USA, 13 pages. https://doi.org/10.1145/3577193.3593712.
* X. Wu, P. Balaprakash, M. Kruse, J. Koo, B. Videau, P. Hovland, V. Taylor, B. Geltz, S. Jana, and M. Hall, "ytopt: Autotuning Scientific Applications for Energy Efficiency at Large Scales", Cray User Group Conference 2023 (CUG’23), Helsinki, Finland, May 7-11, 2023. DOI: [10.48550/arXiv.2303.16245](https://doi.org/10.48550/arXiv.2303.16245)
* X. Wu, M. Kruse, P. Balaprakash, H. Finkel, P. Hovland, V. Taylor, and M. Hall, "Autotuning PolyBench benchmarks with LLVM Clang/Polly loop optimization pragmas using Bayesian optimization (extended version)," Concurrency and Computation. Practice and Experience, Volume 34, Issue 20, 2022. ISSN 1532-0626 DOI: [10.1002/cpe.6683](https://doi.org/10.1002/cpe.6683) 
* J. Koo, P. Balaprakash, M. Kruse, X. Wu, P. Hovland, and M. Hall, "Customized Monte Carlo Tree Search for LLVM/Polly's Composable Loop Optimization Transformations," in Proceedings of 12th IEEE International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS21), pages 82–93, 2021. DOI: [10.1109/PMBS54543.2021.00015](https://scwpub21:conf21%2f%2f@conferences.computer.org/scwpub/pdfs/PMBS2021-vSqRXl4nJSV5KT4jWO5cW/111800a082/111800a082.pdf)
* X. Wu, M. Kruse, P. Balaprakash, H. Finkel, P. Hovland, V. Taylor, and M. Hall, "Autotuning PolyBench Benchmarks with LLVM Clang/Polly Loop Optimization Pragmas Using Bayesian Optimization," in Proceedings of 11th IEEE International Workshop on Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS20), pages 61–70, 2020. DOI: [10.1109/PMBS51919.2020.00012](https://ieeexplore.ieee.org/document/9307884) 
* P. Balaprakash, J. Dongarra, T. Gamblin, M. Hall, J. K. Hollingsworth, B. Norris, and R. Vuduc, "Autotuning in High-Performance Computing Applications," Proceedings of the IEEE, vol. 106, no. 11, 2018. DOI: [10.1109/JPROC.2018.2841200](https://ieeexplore.ieee.org/document/8423171) 
*  T. Nelson, A. Rivera, P. Balaprakash, M. Hall, P. Hovland, E. Jessup, and B. Norris, "Generating efficient tensor contractions for GPUs," in Proceedings of 44th International Conference on Parallel Processing, pages 969–978, 2015. DOI: [10.1109/ICPP.2015.106](https://ieeexplore.ieee.org/document/7349652) 

# Acknowledgements
* SciDAC RAPIDS3, U.S. Department of Energy ASCR (10/2025--present)
* SciDAC RAPIDS and OASIS, U.S. Department of Energy ASCR (1/2024--9/2025)
* PROTEAS-TUNE, U.S. Department of Energy ASCR Exascale Computing Project (2018--2023)
* YTune: Autotuning Compiler Technology for Cross-Architecture Transformation and Code Generation, U.S. Department of Energy Exascale Computing Project (2016--2018) 
* Scalable Data-Efficient Learning for Scientific Domains, U.S. Department of Energy 2018 Early Career Award funded by the Advanced Scientific Computing Research program within the DOE Office of Science (2018--2023)


