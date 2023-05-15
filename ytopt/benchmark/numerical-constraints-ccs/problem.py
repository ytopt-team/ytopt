import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *

import sys
import cconfigspace as CSS
from skopt.space import Real, Integer, Categorical

cs = CCS.ConfigurationSpace(name = "numerical-constraints-ccs")
cs.rng.seed = 1234


p1 = CCS.NumericalParameter.Int(lower = 0, upper = 200, name='x1')
p2 = CCS.NumericalParameter.Int(lower = 0, upper = 200, name='x2', default=1)
p3 = CCS.NumericalParameter.Int(lower = 0, upper = 200, name='x3')

cs.add_parameters([p1, p2, p3])

cs.add_forbidden_clause("x1*x2 + x1*x3 + x2*x3 > 12288")
cs.add_forbidden_clause("(x1 + x2) % 2 == 0")

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
