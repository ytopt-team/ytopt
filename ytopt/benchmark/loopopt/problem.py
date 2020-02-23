import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *

import sys
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from skopt.space import Real, Integer, Categorical


cs = CS.ConfigurationSpace(seed=1234)
p1 = CSH.CategoricalHyperparameter(name='p1', choices=['None', '#pragma omp #p3', '#pragma omp target #p3', '#pragma omp target #p5', '#pragma omp #p4'])
p3 = CSH.CategoricalHyperparameter(name='p3', choices=['None', '#parallel for #p4', '#parallel for #p6', '#parallel for #p7'])
p4 = CSH.CategoricalHyperparameter(name='p4', choices=['None', 'simd'])
p5 = CSH.CategoricalHyperparameter(name='p5', choices=['None', '#dist_schedule static', '#dist_schedule #p11'])
p6 = CSH.CategoricalHyperparameter(name='p6', choices=['None', '#schedule #p10', '#schedule #p11'])
p7 = CSH.CategoricalHyperparameter(name='p7', choices=['None', '#numthreads #p12'])
p10 = CSH.CategoricalHyperparameter(name='p10', choices=['static', 'dynamic'])
p11 = CSH.OrdinalHyperparameter(name='p11', sequence=['1', '8', '16'])
p12 = CSH.OrdinalHyperparameter(name='p12', sequence=['1', '8', '16'])

cs.add_hyperparameters([p1, p3, p4, p5, p6, p7, p10, p11, p12])

#make p3 an active parameter when p1 value is ... 
cond0 = CS.EqualsCondition(p3, p1, '#pragma omp #p3')
cond1 = CS.EqualsCondition(p3, p1, '#pragma omp target #p3')
cond2 = CS.EqualsCondition(p5, p1, '#pragma omp target #p5')
cond3 = CS.EqualsCondition(p4, p1, '#pragma omp #p4')


cond4 = CS.EqualsCondition(p4, p3, '#parallel for #p4')
cond5 = CS.EqualsCondition(p6, p3, '#parallel for #p6')
cond6 = CS.EqualsCondition(p7, p3, '#parallel for #p7')

cond7 = CS.EqualsCondition(p11, p5, '#dist_schedule #p11')

cond8 = CS.EqualsCondition(p10, p6, '#schedule #p10')
cond9 = CS.EqualsCondition(p11, p6, '#schedule #p11')

cond10 = CS.EqualsCondition(p12, p7, '#numthreads #p12')

cs.add_condition(CS.OrConjunction(cond0,cond1))
cs.add_condition(cond2) 
cs.add_condition(CS.OrConjunction(cond3,cond4))
cs.add_condition(cond5) 
cs.add_condition(cond6) 
cs.add_condition(cond8) 
cs.add_condition(CS.OrConjunction(cond7,cond9))
cs.add_condition(cond10)



# problem space
task_space = None

input_space = cs 

output_space = Space([
    Real(-inf, inf, name='y')
])

def myobj(point: dict):
    s = np.random.uniform()
    return s

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
