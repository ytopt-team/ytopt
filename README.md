<p align="center">
<img src="docs/_static/logo/medium.png">
</p>

[![Documentation Status](https://readthedocs.org/projects/ytopt/badge/?version=latest)](https://ytopt.readthedocs.io/en/latest/?badge=latest)

# What is Ytopt/SuRF?

Ytopt/SuRF is a machine-learning-based search software package that consists of sampling a small number of input parameter configurations 
and progressively fitting a surrogate model over the input-output space until exhausting the user-defined time or maximum number of 
evaluations. The package provides two different class of methods: Bayesian Optimization and Reinforcement Learning.
The asynchronous aspect allows the search to avoid waiting for all the evaluation results before proceeding to the next iteration. As 
soon as an evaluation is finished, the data is used to retrain the surrogate model, which is then used to bias the search toward the 
promising configurations. The framework is designed to operate in the master-worker computational paradigm, where one master node fits 
the surrogate model and generates promising input configurations and worker nodes perform the computationally expensive evaluations and 
return the outputs to the master node.

# Documentation

ytopt documentation is on : [ReadTheDocs](https://ytopt.readthedocs.io)

# Directory structure

```
docs/	
    documentation
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
# How do I learn more?

* Documentation: https://ytopt.readthedocs.io

* GitHub repository: https://github.com/ytopt-team/ytopt

# Who is responsible?

The core ytopt team is at Argonne National Laboratory:

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
