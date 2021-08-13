# import required library
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

cs = CS.ConfigurationSpace(seed=1234)
p1 = CSH.CategoricalHyperparameter(name='p1', choices=[' ','#pragma omp #P3','#pragma omp target teams distribute #P3 #P5 #P9 is_device_ptr(A, B)','#pragma omp #P4'], default_value=' ')
p3 = CSH.CategoricalHyperparameter(name='p3', choices=[' ','parallel for #P4 #P8 #P6 #P7'])
p4 = CSH.CategoricalHyperparameter(name='p4', choices=[' ', 'simd'])
p5 = CSH.CategoricalHyperparameter(choices=[' ', 'dist_schedule(static, #P11)'],name='p5') #make a different second param 64 - 512
p6 = CSH.CategoricalHyperparameter(choices=[' ', 'schedule(#P10, #P11)', 'schedule(#P10)'],name='p6')
p7 = CSH.CategoricalHyperparameter(choices=[' ', 'num_threads(#P12)'],name='p7')
p8 = CSH.CategoricalHyperparameter(choices=[' ', 'collapse(2)'],name='p8')
p9 = CSH.CategoricalHyperparameter(choices=[' ', 'thread_limit(#P14)'],name='p9')
p10 = CSH.CategoricalHyperparameter(choices=['static','dynamic'], name='p10')
p11 = CSH.OrdinalHyperparameter(sequence=['1', '2', '4', '8', '16'], name = 'p11') #n(size of data)/num thrads. maybe 2 and 4?
p12 = CSH.OrdinalHyperparameter(sequence=['8',  '16', '32', '64', '72', '128', '176'], name='p12') #need to make it higher, maybe get rid of the low ones
p14 = CSH.OrdinalHyperparameter(sequence=['32', '64', '128', '256'], name='p14')
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
cond11 = CS.EqualsCondition(p14, p9, 'thread_limit(#P14)')

forbidden_clause = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p1, '#pragma omp #P4'), CS.ForbiddenEqualsClause(p4, ' '))
forbidden_clause3 = CS.ForbiddenAndConjunction(CS.ForbiddenEqualsClause(p1, '#pragma omp #P3'), CS.ForbiddenEqualsClause(p3, ' '))

cs.add_forbidden_clauses([forbidden_clause,forbidden_clause3])
cs.add_conditions([cond0, cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond11])

#cs.add_conditions([cond0, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond11])
#in case there is #P2 that needs to be replaced

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
        value = list(point.values())
        print('VALUES:', point)
        params = {k.upper(): v for k, v in point.items()}
        result = obj.findRuntime(value, params)
        return result
    
    x = np.array(list(point.values())) #len(point) = 13 or 26
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
