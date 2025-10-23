import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *
import os
import sys
import time
import json
import math

import os
import sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE)+ '/plopper')
from plopper import Plopper
nparams = 26

cs = CS.ConfigurationSpace(seed=1234)
p1 = CSH.CategoricalHyperparameter(name='p1', choices=[" ","#pragma omp #P3 private(j)","#pragma omp target teams distribute #P3 #P5 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1]) private(j)","#pragma omp #P4"], default_value=" ") 
p2 = CSH.CategoricalHyperparameter(choices=['','#pragma omp #P15 private(j)','#pragma omp target teams distribute #P15 #P17 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1]) private(j)', '#pragma omp #P16'], name='p2') 
p3 = CSH.CategoricalHyperparameter(name='p3', choices=[" ","parallel for #P4 #P6 #P7"])
p4 = CSH.CategoricalHyperparameter(name='p4', choices=[' ', 'simd'])
p5 = CSH.CategoricalHyperparameter(choices=['', 'dist_schedule(static, #P11)'],name='p5')
p6 = CSH.CategoricalHyperparameter(choices=['', 'schedule(#P10, #P11)', 'schedule(#P10)'],name='p6')
p7 = CSH.CategoricalHyperparameter(choices=['', 'num_threads(#P12)'],name='p7')
p8 = CSH.CategoricalHyperparameter(choices=['', 'collapse(#P13)'],name='p8')
p9 = CSH.CategoricalHyperparameter(choices=['', 'thread_limit(#P14)'],name='p9')
p10 = CSH.CategoricalHyperparameter(choices=['static','dynamic'], name='p10')
p11 = CSH.CategoricalHyperparameter(choices=['1', '8', '16'], name = 'p11')
p12 = CSH.CategoricalHyperparameter(choices=['2', '4', '8', '14', '16', '28'], name='p12')
p13 = CSH.CategoricalHyperparameter(choices=['1', '2', '3'], name='p13')
p14 = CSH.CategoricalHyperparameter(choices=['32', '64', '128', '256'], name='p14')
p15= CSH.CategoricalHyperparameter(choices=['','parallel for #P16 #P18 #P19'],name='p15')
p16 = CSH.CategoricalHyperparameter(choices=['', 'simd'],name='p16')
p17= CSH.CategoricalHyperparameter(choices=['', 'dist_schedule(static, #P23)'],name='p17')
p18= CSH.CategoricalHyperparameter(choices=['', 'schedule(#P22, #P23)', 'schedule(#P22)'],name='p18')
p19= CSH.CategoricalHyperparameter(choices=['', 'num_threads(#P24)'],name='p19')
p20 = CSH.CategoricalHyperparameter(choices=['', 'collapse(#P25)'],name='p20')
p21 = CSH.CategoricalHyperparameter(choices=['', 'thread_limit(#P26)'],name='p21')
p22 = CSH.CategoricalHyperparameter(choices=['static','dynamic'], name='p22')
p23 = CSH.CategoricalHyperparameter(choices=['1', '8', '16'], name = 'p23')
p24 = CSH.CategoricalHyperparameter(choices=['2', '4', '8', '14', '16', '28'], name='p24')
p25 = CSH.CategoricalHyperparameter(choices=['1', '2', '3'], name='p25')
p26 = CSH.CategoricalHyperparameter(choices=['32', '64', '128', '256'], name='p26')

cs.add_hyperparameters([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/convolution-2d.c',dir_path)

def myobj(point: dict):

  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = [point['p1'],point['p2'],point['p3'],point['p4'],point['p5'],
          point['p6'], point['p7'],point['p8'],point['p9'],point['p10'],   
          point['p11'],point['p12'],point['p13'],point['p14'],point['p15'],
          point['p16'], point['p17'],point['p18'],point['p19'],point['p20'],   
          point['p21'],point['p22'],point['p23'],point['p24'],point['p25'],
          point['p26']]
    print('VALUES:',
          point['p1'],point['p2'],point['p3'],point['p4'],point['p5'],
          point['p6'], point['p7'],point['p8'],point['p9'],point['p10'],   
          point['p11'],point['p12'],point['p13'],point['p14'],point['p15'],
          point['p16'], point['p17'],point['p18'],point['p19'],point['p20'],   
          point['p21'],point['p22'],point['p23'],point['p24'],point['p25'],
          point['p26'])
    params = ["P1","P2","P3","P4","P5","P6","P7","P8","P9","P10",
              "P11","P12","P13","P14","P15","P16","P17","P18","P19","P20",
              "P21","P22","P23","P24","P25","P26"]
    result = obj.findRuntime(value, params)
    return result

  x = np.array([point[f'p{i}'] for i in range(1, len(point))])
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

if __name__ == '__main__':
    params = ["P1", "P2", "P3","P4"]
    x = ['#pragma omp parallel schedule(#P2)', 'static', '8','1']
    obj = Plopper(dir_path+'/convolution-2d.c',dir_path)
    retVal = obj.findRuntime(x, params)
    print(retVal)
