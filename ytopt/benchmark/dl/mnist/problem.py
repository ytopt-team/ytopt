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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from ytopt.benchmark.plopper.plublic_plopper import PyPlopper
nparams = 4

cs = CS.ConfigurationSpace(seed=1234)
#batch_size
p0= CSH.OrdinalHyperparameter(name='p0', sequence=['16','32','64','100','128','200','256','300','400','512'], default_value='128')
#epochs
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['1','2','4','8','12','16','20','22','24','30'], default_value='20')
#dropout rate
p2= CSH.OrdinalHyperparameter(name='p2', sequence=['0.1', '0.15', '0.2', '0.25','0.4'], default_value='0.2')
#optimizer
p3= CSH.CategoricalHyperparameter(name='p3', choices=['rmsprop','adam','sgd','adamax','adadelta','adagrad','nadam'], default_value='rmsprop')

cs.add_hyperparameters([p0, p1, p2, p3])

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

x1=['p0','p1','p2','p3']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]]]
    print('VALUES:',point[x1[0]])
    params = ["P1","P2","P3","P4"]

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
