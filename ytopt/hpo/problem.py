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
nparams = 4

cs = CS.ConfigurationSpace(seed=1234)
#batch_size
p0 = CSH.UniformIntegerHyperparameter(name='p0', lower=256, upper=20000, default_value=20000, q=8)
#epochs
p1= CSH.UniformIntegerHyperparameter(name='p1', lower=100, upper=500, default_value=100, q=10)
#learning rate
p2= CSH.UniformFloatHyperparameter(name='p2', lower=0.000001, upper=0.1, q=0.000001, default_value=0.0005)
#dropout rate
p3= CSH.UniformFloatHyperparameter(name='p3', lower=0.0, upper=0.5, q=0.01, default_value=0.2)
#optimizer
p4= CSH.CategoricalHyperparameter(name='p4', choices=['RMSprop','Adam','SGD'], default_value='Adam')
#L2 Weight Decay
p5= CSH.UniformFloatHyperparameter(name='p5', lower=0.000001, upper=0.01, q=0.000001, default_value=0.0001)
# Weight Initialization
p6= CSH.CategoricalHyperparameter(name='p6', choices=['xavier','he','uniform'], default_value='xavier')
# Activation Functions L1
p7= CSH.CategoricalHyperparameter(name='p7', choices=['tanh','sigmoid','ELU','SiLU','softmax'], default_value='tanh')
# Activation Functions L2
p8= CSH.CategoricalHyperparameter(name='p8', choices=['tanh','sigmoid','ELU','SiLU','softmax'], default_value='tanh')
# Activation Functions L3
p9= CSH.CategoricalHyperparameter(name='p9', choices=['tanh','sigmoid','ELU','SiLU','softmax'], default_value='ELU')
# Activation Functions L4
p10= CSH.CategoricalHyperparameter(name='p10', choices=['tanh','sigmoid','ELU','SiLU','softmax'], default_value='SiLU')
# Activation Functions L5
p11= CSH.CategoricalHyperparameter(name='p11', choices=['tanh','sigmoid','ELU','SiLU','softmax'], default_value='tanh')
#number of nodes L1
p12= CSH.UniformIntegerHyperparameter(name='p12', lower=400, upper=1000, default_value=800, q=10)
#number of nodes L2
p13= CSH.UniformIntegerHyperparameter(name='p13', lower=100, upper=400, default_value=200, q=10)
#number of nodes L3
p14= CSH.UniformIntegerHyperparameter(name='p14', lower=40, upper=100, default_value=40, q=10)
#number of nodes L4
p15= CSH.UniformIntegerHyperparameter(name='p15', lower=20, upper=40, default_value=20, q=2)
#number of nodes L5
p16= CSH.UniformIntegerHyperparameter(name='p16', lower=2, upper=20, default_value=10, q=2)

cs.add_hyperparameters([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16])

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

x1=['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10','p11','p12','p13','p14','p15','p16']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]],point[x1[5]],point[x1[6]],point[x1[7]],point[x1[8]],point[x1[9]],point[x1[10]],point[x1[11]],point[x1[12]],point[x1[13]],point[x1[14]],point[x1[15]],point[x1[16]]]
    print('VALUES:',point[x1[0]])
    params = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14","P15","P16"]

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
