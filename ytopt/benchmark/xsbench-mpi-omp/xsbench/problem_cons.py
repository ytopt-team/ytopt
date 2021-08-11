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
# number of threads
p0= CSH.UniformIntegerHyperparameter(name='p0', lower=4, upper=8, default_value=8)
# choice for openmp static/dynamic schedule
p1 = CSH.CategoricalHyperparameter(name='p1', choices=['dynamic,#P3','guided,#P3','static,#P3','auto'], default_value='dynamic,#P3')
#omp parallel
p2= CSH.CategoricalHyperparameter(name='p2', choices=["#pragma omp parallel for", " "], default_value=' ')
#block size for openmp static/dynamic schedule
p3= CSH.OrdinalHyperparameter(name='p3', sequence=['10','20','40','64','80','100','128','160','200'], default_value='100')
cs.add_hyperparameters([p0, p1, p2, p3])

# add condition
cond1 = CS.InCondition(p3, p1, ['dynamic,#P3','static,#P3','guided,#P3'])
cs.add_conditions([cond1])

# problem space
task_space = None
input_space = cs
output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/mmp_cons.c',dir_path)

x1=['p0','p1','p2','p3']
def myobj(point: dict):
    def plopper_func(x):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        value = [point[x1[0]],point[x1[1]],point[x1[2]],point[x1[3]]]
        print('CONFIG:',point)
        params = ["P0","P1","P2","P3"]
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
