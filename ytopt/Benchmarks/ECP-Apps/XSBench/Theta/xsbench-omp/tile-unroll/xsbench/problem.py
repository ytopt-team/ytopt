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
from plopper import Plopper

cs = CS.ConfigurationSpace(seed=1234)
# number of threads
p0= CSH.OrdinalHyperparameter(name='p0', sequence=['4','8','16','32','48','64','96','128','192','256'], default_value='64')
#block size for openmp dynamic schedule
p1= CSH.OrdinalHyperparameter(name='p1', sequence=['10','20','40','50','64','80','100','128','160','200','256','400'], default_value='100')
#clang unrolling
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma clang loop unrolling full", " "], default_value=' ')
#omp parallel
p3= CSH.CategoricalHyperparameter(name='p3', choices=["#pragma omp parallel for", " "], default_value=' ')
# tile size for one dimension for 2D tiling
p4= CSH.OrdinalHyperparameter(name='p4', sequence=['2','4','8','16','32','64','96','128','256','512','1024'], default_value='96')
# tile size for another dimension for 2D tiling
p5= CSH.OrdinalHyperparameter(name='p5', sequence=['2','4','8','16','32','64','96','128','256','512','1024'], default_value='256')
# omp placement
p6= CSH.CategoricalHyperparameter(name='p6', choices=['cores','threads','sockets'], default_value='cores')
# OMP_PROC_BIND
p7= CSH.CategoricalHyperparameter(name='p7', choices=['close','spread','master'], default_value='close')
# OMP_SCHEDULE
p8= CSH.CategoricalHyperparameter(name='p8', choices=['dynamic','static'], default_value='static')

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp.c',dir_path)

x1=['p0','p1','p2','p3','p4','p5','p6','p7','p8']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]],point[x1[5]],point[x1[6]],point[x1[7]],point[x1[8]]]
    print('VALUES:',point[x1[0]])
    os.system("processexe.pl exe.pl " +point[x1[0]])
    params = ["P0","P1","P2","P3","P4","P5","P6","P7","P8"]

    result = obj.findRuntime(value, params)
    return result

  x = np.array([point[f'p{i}'] for i in range(len(point))])
  results = plopper_func(x)
  print('OUTPUT:%f',results)

  return results

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
