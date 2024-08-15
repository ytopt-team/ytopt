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

from plopper import Plopper

cs = CS.ConfigurationSpace(seed=1234)
#mixed ratio
p0= CSH.UniformFloatHyperparameter(name='p0', lower=0.0, upper=1.0, q=0.0001, log = False)
#sigmoid_ratio: float = 0.0001,
p1= CSH.UniformFloatHyperparameter(name='p1', lower=0.00005, upper=0.0003, default_value=0.0001, log = False)
#gaussian_ratio: float = 1
p2= CSH.UniformFloatHyperparameter(name='p2', lower=0.8, upper=1, default_value=1, log = False)

cs.add_hyperparameters([p0,p1,p2])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/dlp.py',dir_path)

x1=['p0','p1','p2']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]]]
    print('VALUES:',point[x1[0]])
    params = ["P0","P1","P2"]

    result = obj.findRuntime(value, params)
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT: ',results)

  return results

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
