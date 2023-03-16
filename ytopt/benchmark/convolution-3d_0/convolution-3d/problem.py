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

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/convolution-3d.c',dir_path)

cs = CS.ConfigurationSpace(seed=1234)
p1 = CSH.CategoricalHyperparameter(name='p1', choices=[' ','#pragma omp #P3','#pragma omp target teams distribute #P3 #P5 #P9 is_device_ptr(A, B)','#pragma omp #P4'], default_value=' ')
p3 = CSH.CategoricalHyperparameter(name='p3', choices=[' ','parallel for #P4 #P8 #P6 #P7'])
p4 = CSH.CategoricalHyperparameter(name='p4', choices=[' ', 'simd'])
p5 = CSH.CategoricalHyperparameter(choices=[' ', 'dist_schedule(static, #P11)'],name='p5') #make a different second param 64 - 512
p6 = CSH.CategoricalHyperparameter(choices=[' ', 'schedule(#P10, #P11)', 'schedule(#P10)'],name='p6')
p7 = CSH.CategoricalHyperparameter(choices=[' ', 'num_threads(#P12)'],name='p7')
p8 = CSH.CategoricalHyperparameter(choices=[' ', 'collapse(2)', 'collapse(3)'],name='p8')
p9 = CSH.CategoricalHyperparameter(choices=[' ', 'thread_limit(#P14)'],name='p9')
p10 = CSH.CategoricalHyperparameter(choices=['static','dynamic'], name='p10')
#p11 = CSH.UniformIntegerHyperparameter(name='p11', lower=1, upper=512) #[1, 512]
p11 = CSH.OrdinalHyperparameter(sequence=['1', '2', '4', '8', '16'], name = 'p11') #n(size of data)/num thrads. maybe 2 and 4?
#p12 = CSH.UniformIntegerHyperparameter(name='p12', lower=18, upper=96)
p12 = CSH.OrdinalHyperparameter(sequence=['8',  '16', '32', '64', '72', '128', '176'], name='p12') #need to make it higher, maybe get rid of the low ones
#p13 = CSH.CategoricalHyperparasdmeter(choices=['1', '2'], name='p13') #modified to be only 2 for now.
p14 = CSH.OrdinalHyperparameter(sequence=['32', '64', '128', '256'], name='p14')


#p11, 12, 14 #should be ordinal

#p1 always there, so add it, then check if p2 exists
cs.add_hyperparameters([p1,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p14])
cond0 = CS.InCondition(p3, p1, ['#pragma omp #P3','#pragma omp target teams distribute #P3 #P5 #P9 is_device_ptr(A, B)'])
cond1 = CS.EqualsCondition(p5, p1, '#pragma omp target teams distribute #P3 #P5 #P9 is_device_ptr(A, B)')
cond2 = CS.EqualsCondition(p9, p1, '#pragma omp target teams distribute #P3 #P5 #P9 is_device_ptr(A, B)')

cond3 = CS.OrConjunction(CS.EqualsCondition(p4, p1, '#pragma omp #P4'),
                         CS.EqualsCondition(p4, p3, 'parallel for #P4 #P8 #P6 #P7'))

cond4 = CS.EqualsCondition(p6, p3, 'parallel for #P4 #P8 #P6 #P7')
cond5 = CS.EqualsCondition(p7, p3, 'parallel for #P4 #P8 #P6 #P7')
cond6 = CS.EqualsCondition(p8, p3, 'parallel for #P4 #P8 #P6 #P7')

cond7 = CS.InCondition(p10, p6, ['schedule(#P10, #P11)', 'schedule(#P10)'])

cond8 = CS.OrConjunction(CS.EqualsCondition(p11, p5, 'dist_schedule(static, #P11)'),
                          CS.EqualsCondition(p11, p6, 'schedule(#P10, #P11)'))
cond9 = CS.EqualsCondition(p12, p7, 'num_threads(#P12)')

#cond10 = CS.EqualsCondition(p13, p8, 'collapse(#P13)')

cond11 = CS.EqualsCondition(p14, p9, 'thread_limit(#P14)')

forbidden_clause = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p1, '#pragma omp #P4'), CS.ForbiddenEqualsClause(p4, ' '))
forbidden_clause3 = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p1, '#pragma omp #P3'), CS.ForbiddenEqualsClause(p3, ' '))

cs.add_forbidden_clauses([forbidden_clause,forbidden_clause3])

cs.add_conditions([cond0, cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond11])

#in case there is #P2 that needs to be replaced
if (obj.p2check(dir_path+'/convolution-3d.c')): #found p2
  print("found p2")
  p2 = CSH.CategoricalHyperparameter(choices=[' ','#pragma omp #P15','#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:(nx-1)*(ny-1)]) map(from:B[0:(nx-1)*(ny-1)])', '#pragma omp #P16'], name='p2')
  p15 = CSH.CategoricalHyperparameter(choices=[' ', 'parallel for #P16 #P20 #P18 #P19'],name='p15')
  p16 = CSH.CategoricalHyperparameter(choices=[' ', 'simd'],name='p16')
  p17 = CSH.CategoricalHyperparameter(choices=[' ', 'dist_schedule(static, #P23)'],name='p17')
  p18 = CSH.CategoricalHyperparameter(choices=[' ', 'schedule(#P22, #P23)', 'schedule(#P22)'],name='p18')
  p19 = CSH.CategoricalHyperparameter(choices=[' ', 'num_threads(#P24)'],name='p19')
  p20 = CSH.CategoricalHyperparameter(choices=[' ', 'collapse(2)'],name='p20')
  p21 = CSH.CategoricalHyperparameter(choices=[' ', 'thread_limit(#P26)'],name='p21')
  p22 = CSH.CategoricalHyperparameter(choices=['static', 'dynamic'], name='p22')
  #p23 = CSH.UniformIntegerHyperparameter(lower=1, upper = 512, name='p23') #[1, 512]
  p23 = CSH.CategoricalHyperparameter(choices=['1', '8', '16'], name = 'p23')
  p24 = CSH.CategoricalHyperparameter(choices=['2', '4', '8', '14', '16', '28'], name='p24')
  #p25 = CSH.CategoricalHyperparameter(choices=['1', '2', '3'], name='p25')
  p26 = CSH.CategoricalHyperparameter(choices=['32', '64', '128', '256'], name='p26')

  cs.add_hyperparameters([p2,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p26])
  cond12 = CS.InCondition(p15, p2, ['#pragma omp #P15', '#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:(nx-1)*(ny-1)]) map(from:B[0:(nx-1)*(ny-1)])'])
  cond13 = CS.EqualsCondition(p17, p2, '#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:(nx-1)*(ny-1)]) map(from:B[0:(nx-1)*(ny-1)])')
  cond14 = CS.EqualsCondition(p21, p2, '#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:(nx-1)*(ny-1)]) map(from:B[0:(nx-1)*(ny-1)])')

  cond15 = CS.OrConjunction(CS.EqualsCondition(p16, p2, '#pragma omp #P16'),
                            CS.EqualsCondition(p16, p15, 'parallel for #P16 #P20 #P18 #P19'))
                            
  cond16 = CS.EqualsCondition(p18, p15, 'parallel for #P16 #P20 #P18 #P19')
  cond17 = CS.EqualsCondition(p19, p15, 'parallel for #P16 #P20 #P18 #P19')
  cond18 = CS.EqualsCondition(p20, p15, 'parallel for #P16 #P20 #P18 #P19')

  cond19 = CS.InCondition(p22, p18, ['schedule(#P22, #P23)', 'schedule(#P22)'])

  cond20 = CS.OrConjunction(CS.EqualsCondition(p23, p17, 'dist_schedule(static, #P23)'),
                            CS.EqualsCondition(p23, p18, 'schedule(#P22, #P23)'))
  cond21 = CS.EqualsCondition(p24, p19, 'num_threads(#P24)')

  #cond22 = CS.EqualsCondition(p25, p20, 'collapse(#P25)')

  cond23 = CS.EqualsCondition(p26, p21, 'thread_limit(#P26)')

  forbidden_clause2 = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p2, '#pragma omp #P16'), CS.ForbiddenEqualsClause(p16, ' '))
  forbidden_clause4 = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p2, '#pragma omp #P15'), CS.ForbiddenEqualsClause(p15, ' '))

  cs.add_forbidden_clauses([forbidden_clause2, forbidden_clause4])
  cs.add_conditions([cond12, cond13, cond14, cond15, cond16, cond17, cond18, cond19, cond20, cond21, cond23])

# problem space
task_space = None
input_space = cs
output_space = Space([
     Real(0.0, inf, name="time")
])

def myobj(point: dict):
  
  def plopper_func(x):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    value = list(point.values())
    print('VALUES:', point)
    params = {k.upper(): v for k, v in point.items()}
    result = obj.findRuntime(value, params)
    return result

  x = np.array(list(point.values())) #len(point) = 13 or 26
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

if __name__ == '__main__':
    params = ["P1", "P2", "P3","P4"]
    x = ['#pragma omp parallel schedule(#P2)', 'static', '8','1']
    obj = Plopper(dir_path+'/convolution-3d.c',dir_path)
    retVal = obj.findRuntime(x, params)
    print(retVal)
