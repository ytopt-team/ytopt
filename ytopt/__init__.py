"""
Ytopt is a machine-learning-based search software package that consists of sampling
a small number of input parameter configurations, evaluating them, and progressively
fitting a surrogate model over the input-output space until exhausting the user-defined
time or maximum number of evaluations. The package provides two different class of methods:
Bayesian Optimization and Reinforcement Learning. The software is designed to operate in the
master-worker computational paradigm, where one master node fits the surrogate model and
generates promising input configurations and worker nodes perform the computationally expensive
 evaluations and return the outputs to the master node. The asynchronous aspect of the
 search allows the search to avoid waiting for all the evaluation results before proceeding
  to the next iteration. As soon as an evaluation is finished, the data is used to retrain
  the surrogate model, which is then used to bias the search toward the promising configurations.
"""

from ytopt.__version__ import __version__
name = 'ytopt'
version = __version__