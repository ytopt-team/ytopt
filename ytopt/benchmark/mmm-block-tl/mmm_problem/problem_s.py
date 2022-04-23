import numpy as np
from autotune import TuningProblem
from autotune.space import *
import os, sys, time, json, math
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper

# create an object of ConfigSpace
cs = CS.ConfigurationSpace(seed=1234)
#block size for openmp dynamic schedule
# p0= CSH.OrdinalHyperparameter(name='BLOCK_SIZE', sequence=['1','2','3','4','5','6','7','8','9','10'], default_value='5')
p0= CSH.UniformIntegerHyperparameter(name='BLOCK_SIZE', lower=1, upper=100, default_value=5)
cs.add_hyperparameters([p0])

# problem space
task_space = None
input_space = cs
output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmm_block_100.cpp',dir_path)

x1=['BLOCK_SIZE']
def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]]]
        print('CONFIG:',point)
        params = ["BLOCK_SIZE"]
        result = obj.findRuntime(value, params, '/mmm_block_100.cpp')
        return result

    x = np.array([point['BLOCK_SIZE']])
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
