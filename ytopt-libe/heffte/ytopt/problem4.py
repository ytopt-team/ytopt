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
from ConfigSpace import ConfigurationSpace, EqualsCondition
from skopt.space import Real, Integer, Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))
from plopper import Plopper

cs = CS.ConfigurationSpace(seed=1234)
# arg1  precision
p0 = CSH.CategoricalHyperparameter(name='p0', choices=["double", "float"], default_value="float")
# arg2  3D array dimension size
p1 = CSH.OrdinalHyperparameter(name='p1', sequence=[64,128,256,512,1024], default_value=128)
# arg3  reorder
p2 = CSH.CategoricalHyperparameter(name='p2', choices=["-no-reorder", "-reorder"," "], default_value=" ")
# arg4 alltoall
p3 = CSH.CategoricalHyperparameter(name='p3', choices=["-a2a", "-a2av", " "], default_value=" ")
# arg5 p2p
p4 = CSH.CategoricalHyperparameter(name='p4', choices=["-p2p", "-p2p_pl"," "], default_value=" ")
# arg6 reshape logic
p5 = CSH.CategoricalHyperparameter(name='p5', choices=["-pencils", "-slabs"," "], default_value=" ")
# arg7
p6 = CSH.CategoricalHyperparameter(name='p6', choices=["-r2c_dir 0", "-r2c_dir 1","-r2c_dir 2", " "], default_value=" ")
# arg8
p7 = CSH.CategoricalHyperparameter(name='p7', choices=["-ingrid 4 1 1", "-ingrid 2 2 1", "-ingrid 2 1 2","-ingrid 1 2 2", " "], default_value=" ")
# arg9
p8 = CSH.CategoricalHyperparameter(name='p8', choices=["-outgrid 4 1 1", "-outgrid 2 2 1", "-outgrid 2 1 2","-outgrid 1 2 2"," "], default_value=" ")
#number of threads
p9= CSH.UniformIntegerHyperparameter(name='p9', lower=2, upper=8, default_value=8, q=2)
#gpu-aware
p10 = CSH.CategoricalHyperparameter(name='p10', choices=["-no-gpu-aware", "-gpu-aware"], default_value="-gpu-aware")

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/speed3d.sh',dir_path)

x1=['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9','p10']

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    #if str(point[x1[3]])=='nan':
        #point[x1[3]]=' '
    value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]],point[x1[4]],point[x1[5]],point[x1[6]],point[x1[7]],point[x1[8]],point[x1[9]],point[x1[10]]]
    print('CONFIG:',point)
    #print(point[x1[4]])
    os.system('./processexe.pl exe.pl ')
    #execmd = './processexe.pl exe.pl ' + str(point[x1[4]])+ ' '+ str(point[x1[5]])+ ' '+ str(point[x1[6]])
    #print(execmd)
    #os.system(execmd)
    os.environ["OMP_NUM_THREADS"] = str(point[x1[9]])
    params = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9","P10"]

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
