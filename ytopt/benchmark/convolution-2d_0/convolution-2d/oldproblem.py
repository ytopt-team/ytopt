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
p1 = CSH.CategoricalHyperparameter(name='p1', choices=[' ','#pragma omp #P3','#pragma omp target teams distribute #P3 #P5 #P9 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])','#pragma omp #P4'], default_value=' ')
p2 = CSH.CategoricalHyperparameter(choices=[' ','#pragma omp #P15','#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])', '#pragma omp #P16'], name='p2')
p3 = CSH.CategoricalHyperparameter(name='p3', choices=[' ','parallel for #P4 #P8 #P6 #P7'])
p4 = CSH.CategoricalHyperparameter(name='p4', choices=[' ', 'simd'])
p5 = CSH.CategoricalHyperparameter(choices=[' ', 'dist_schedule(static, #P11)'],name='p5')
p6 = CSH.CategoricalHyperparameter(choices=[' ', 'schedule(#P10, #P11)', 'schedule(#P10)'],name='p6')
p7 = CSH.CategoricalHyperparameter(choices=[' ', 'num_threads(#P12)'],name='p7')
p8 = CSH.CategoricalHyperparameter(choices=[' ', 'collapse(#P13)'],name='p8')
p9 = CSH.CategoricalHyperparameter(choices=[' ', 'thread_limit(#P14)'],name='p9')
p10 = CSH.CategoricalHyperparameter(choices=['static','dynamic'], name='p10')
p11 = CSH.CategoricalHyperparameter(choices=['1', '8', '16'], name = 'p11')
p12 = CSH.CategoricalHyperparameter(choices=['2', '4', '8', '14', '16', '28', '32', '40', '60', '80'], name='p12')
p13 = CSH.CategoricalHyperparameter(choices=['1', '2'], name='p13') #modified to be only 2 for now.
p14 = CSH.CategoricalHyperparameter(choices=['32', '64', '128', '256'], name='p14')
p15= CSH.CategoricalHyperparameter(choices=[' ','parallel for #P16 #p20 #P18 #P19'],name='p15')
p16 = CSH.CategoricalHyperparameter(choices=[' ', 'simd'],name='p16')
p17= CSH.CategoricalHyperparameter(choices=[' ', 'dist_schedule(static, #P23)'],name='p17')
p18= CSH.CategoricalHyperparameter(choices=[' ', 'schedule(#P22, #P23)', 'schedule(#P22)'],name='p18')
p19= CSH.CategoricalHyperparameter(choices=[' ', 'num_threads(#P24)'],name='p19')
p20 = CSH.CategoricalHyperparameter(choices=[' ', 'collapse(#P25)'],name='p20')
p21 = CSH.CategoricalHyperparameter(choices=[' ', 'thread_limit(#P26)'],name='p21')
p22 = CSH.CategoricalHyperparameter(choices=['static','dynamic'], name='p22')
p23 = CSH.CategoricalHyperparameter(choices=['1', '8', '16'], name = 'p23')
p24 = CSH.CategoricalHyperparameter(choices=['2', '4', '8', '14', '16', '28'], name='p24')
p25 = CSH.CategoricalHyperparameter(choices=['1', '2', '3'], name='p25')
p26 = CSH.CategoricalHyperparameter(choices=['32', '64', '128', '256'], name='p26')

cs.add_hyperparameters([p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26])

cond0 = CS.InCondition(p3, p1, ['#pragma omp #P3','#pragma omp target teams distribute #P3 #P5 #P9 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])'])
cond1 = CS.EqualsCondition(p5, p1, '#pragma omp target teams distribute #P3 #P5 #P9 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])')
cond2 = CS.EqualsCondition(p9, p1, '#pragma omp target teams distribute #P3 #P5 #P9 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])')
cond3 = CS.EqualsCondition(p4, p1, '#pragma omp #P4')

cond4 = CS.InCondition(p15, p2, ['#pragma omp #P15', '#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])'])
cond5 = CS.EqualsCondition(p17, p2, '#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])')
cond6 = CS.EqualsCondition(p21, p2, '#pragma omp target teams distribute #P15 #P17 #P21 map(to:A[0:ni-1][0:nj-1]) map(from:B[0:ni-1][0:nj-1])')

cond7 = CS.EqualsCondition(p16, p2, '#pragma omp #P16')

cond8 = CS.EqualsCondition(p4, p3, 'parallel for #P4 #P8 #P6 #P7')
cond9 = CS.EqualsCondition(p6, p3, 'parallel for #P4 #P8 #P6 #P7')
cond10 = CS.EqualsCondition(p7, p3, 'parallel for #P4 #P8 #P6 #P7')
cond11 = CS.EqualsCondition(p8, p3, 'parallel for #P4 #P8 #P6 #P7')

cond12 = CS.EqualsCondition(p11, p5, 'dist_schedule(static, #P11)')

cond13 = CS.InCondition(p10, p6, ['schedule(#P10, #P11)', 'schedule(#P10)'])
cond14 = CS.EqualsCondition(p11, p6, 'schedule(#P10, #P11)')

cond15 = CS.EqualsCondition(p12, p7, 'num_threads(#P12)')

cond16 = CS.EqualsCondition(p13, p8, 'collapse(#P13)')

cond17 = CS.EqualsCondition(p14, p9, 'thread_limit(#P14)')

cond18 = CS.EqualsCondition(p16, p15, 'parallel for #P16 #p20 #P18 #P19')
cond19 = CS.EqualsCondition(p18, p15, 'parallel for #P16 #p20 #P18 #P19')
cond20 = CS.EqualsCondition(p19, p15, 'parallel for #P16 #p20 #P18 #P19')

cond21 = CS.EqualsCondition(p23, p17, 'dist_schedule(static, #P23)')

cond22 = CS.InCondition(p22, p18, ['schedule(#P22, #P23)', 'schedule(#P22)'])
cond23 = CS.EqualsCondition(p23, p18, 'schedule(#P22, #P23)')

cond24 = CS.EqualsCondition(p24, p19, 'num_threads(#P24)')

cond25 = CS.EqualsCondition(p25, p20, 'collapse(#P25)')

cond26 = CS.EqualsCondition(p26, p21, 'thread_limit(#P26)')

#avoid #pragma omp ' '
forbidden_clause = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p1, '#pragma omp #P4'), CS.ForbiddenEqualsClause(p4, ' '))
forbidden_clause2 = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p2, '#pragma omp #P16'), CS.ForbiddenEqualsClause(p16, ' '))

#avoid #pragma omp ' '
forbidden_clause3 = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p1, '#pragma omp #P3'), CS.ForbiddenEqualsClause(p3, ' '))
forbidden_clause4 = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p2, '#pragma omp #P15'), CS.ForbiddenEqualsClause(p15, ' '))

cs.add_condition(cond0)
cs.add_condition(CS.OrConjunction(cond3,cond8))
cs.add_condition(CS.OrConjunction(cond12,cond14))
cs.add_condition(CS.OrConjunction(cond7,cond18))
cs.add_condition(CS.OrConjunction(cond21,cond23))
cs.add_condition(cond1)
cs.add_condition(cond2)
cs.add_condition(cond4)
cs.add_condition(cond5)
cs.add_condition(cond6)
cs.add_condition(cond9)
cs.add_condition(cond10)
cs.add_condition(cond11)
cs.add_condition(cond13)
cs.add_condition(cond15)
cs.add_condition(cond16)
cs.add_condition(cond17)
cs.add_condition(cond19)
cs.add_condition(cond20)
cs.add_condition(cond22)
cs.add_condition(cond24)
cs.add_condition(cond25)
cs.add_condition(cond26)
cs.add_forbidden_clause(forbidden_clause)
cs.add_forbidden_clause(forbidden_clause2)
cs.add_forbidden_clause(forbidden_clause3)
cs.add_forbidden_clause(forbidden_clause4)


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
    value = [point[f'p{i}'] for i in range(1, len(point)+1)]         
    print('VALUES:', point)
    params = [f"P{i}" for i in range(1, len(point)+1)]
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

