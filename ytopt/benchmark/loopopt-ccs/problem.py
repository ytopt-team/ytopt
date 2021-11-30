import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *

import sys
import cconfigspace as CSS
from skopt.space import Real, Integer, Categorical

cs = CCS.ConfigurationSpace(name = "loopopt-ccs")
cs.rng.seed = 1234


p1  = CCS.CategoricalHyperparameter(name='p1',  values=['None', '#pragma omp #p3', '#pragma omp target #p3', '#pragma omp target #p5', '#pragma omp #p4'])
p3  = CCS.CategoricalHyperparameter(name='p3',  values=['None', '#parallel for #p4', '#parallel for #p6', '#parallel for #p7'])
p4  = CCS.CategoricalHyperparameter(name='p4',  values=['None', 'simd'])
p5  = CCS.CategoricalHyperparameter(name='p5',  values=['None', '#dist_schedule static', '#dist_schedule #p11'])
p6  = CCS.CategoricalHyperparameter(name='p6',  values=['None', '#schedule #p10', '#schedule #p11'])
p7  = CCS.CategoricalHyperparameter(name='p7',  values=['None', '#numthreads #p12'])
p10 = CCS.CategoricalHyperparameter(name='p10', values=['static', 'dynamic'])
p11 = CCS.OrdinalHyperparameter(name='p11', values=['1', '8', '16'])
p12 = CCS.OrdinalHyperparameter(name='p12', values=['1', '8', '16'])

cs.add_hyperparameters([p1, p3, p4, p5, p6, p7, p10, p11, p12])

#make p3 an active parameter when p1 value is ...
cs.set_condition(p3,  "p1 # ['#pragma omp #p3', '#pragma omp target #p3']")
cs.set_condition(p5,  "p1 == '#pragma omp target #p5'")
cs.set_condition(p4,  "p1 == '#pragma omp #p4' || p3 == '#parallel for #p4'")
cs.set_condition(p6,  "p3 == '#parallel for #p6'")
cs.set_condition(p7,  "p3 == '#parallel for #p7'")
cs.set_condition(p11, "p5 == '#dist_schedule #p11' || p6 == '#schedule #p11'")
cs.set_condition(p10, "p6 == '#schedule #p10'")
cs.set_condition(p12, "p7 == '#numthreads #p12'")

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
