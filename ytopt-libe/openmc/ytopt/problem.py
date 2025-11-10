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
from ConfigSpace import ConfigurationSpace, EqualsCondition, InCondition, LessThanCondition
from skopt.space import Real, Integer, Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))
from plopper import Plopper

cs = CS.ConfigurationSpace(seed=1234)
# queuing logic type
p0 = CSH.CategoricalHyperparameter(name='p0', choices=["openmc", "openmc-queueless"], default_value="openmc-queueless")
# maximum number of particles in-flight
p1= CSH.UniformIntegerHyperparameter(name='p1', lower=100000, upper=8000000, default_value=1000000, q=100000)
# number of logarithmic hash grid bins
p2 = CSH.UniformIntegerHyperparameter(name='p2', lower=1000, upper=100000, default_value=4000, q=1000)
# minimum sorting threshold
p3 = CSH.UniformIntegerHyperparameter(name='p3', lower=0, upper=100000, default_value=20000, q=10000) 
#number of threads
p4= CSH.UniformIntegerHyperparameter(name='p4', lower=2, upper=64, default_value=64, q=2)
#number of tasks per gpu
p5= CSH.OrdinalHyperparameter(name='p5',sequence=[1, 2], default_value=1)
#thread binding
p6= CSH.CategoricalHyperparameter(name='p6', choices=['cores','threads','sockets'], default_value='threads')

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6])
#cond = EqualsCondition(p3, p0, "openmc")
cond = InCondition(p3, p0, ["openmc"])
#cond1 = LessThanCondition(p5,p4,64)
cs.add_conditions([cond])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/openmc.sh',dir_path)

x1=['p0','p1','p2','p3','p4','p5','p6']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
#    x[np.isnan(x)] = 0
#    if str(point[x1[3]])=='nan':
#        point[x1[3]]= 0
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]],point[x1[5]],point[x1[6]]]
    print('CONFIG:',point)
    #print(point[x1[4]])
    #os.system('./processexe.pl exe.pl ' + str(point[x1[4]]))
    execmd = './processexe.pl exe.pl ' + str(point[x1[4]])+ ' '+ str(point[x1[5]])+ ' '+ str(point[x1[6]])
    #print(execmd)
    os.system(execmd)
    os.environ["OMP_NUM_THREADS"] = str(point[x1[4]])
    params = ["P0","P1","P2","P3","P4","P5","P6"]

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
