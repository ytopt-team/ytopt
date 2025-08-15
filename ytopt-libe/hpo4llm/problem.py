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

#HERE = os.path.dirname(os.path.abspath(__file__))
#sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
# from ytopt.benchmark.plopper.plublic_plopper import PyPlopper
from .plopper import Plopper as PyPlopper
nparams = 5

cs = CS.ConfigurationSpace(seed=1234)
#Entropy
p0= CSH.UniformFloatHyperparameter(name='p0', lower=0.0, upper=2.0, default_value=1.00, q=0.01)
# A
p1= CSH.UniformFloatHyperparameter(name='p1', lower=100.0, upper=1000.0, default_value=550.0, q=0.1)
# B
p2= CSH.UniformFloatHyperparameter(name='p2', lower=100.0, upper=1000.0, default_value=550.0, q=0.1)
#alpha
p3= CSH.UniformFloatHyperparameter(name='p3', lower=0.1, upper=1.0, default_value=0.5, q=0.01)
#beta
p4= CSH.UniformFloatHyperparameter(name='p4', lower=0.1, upper=1.0, default_value=0.5, q=0.01)

cs.add_hyperparameters([p0, p1, p2, p3, p4])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = PyPlopper(dir_path+'/dlp.py',dir_path)

x1=['p0','p1','p2','p3','p4']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]]]
    params = ["P1","P2","P3","P4","P5"]
    result = obj.findRuntime(value, params)
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT: ',results)

  return results
print("Calling myobj with point:", input_space.sample_configuration().get_dictionary())
Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
