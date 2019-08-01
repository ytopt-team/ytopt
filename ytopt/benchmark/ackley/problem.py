import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *

# problem space
task_space = None

input_space = Space([
    Real(-15, 15, name=f'x_{i}') for i in range(10)
])

output_space = Space([
    Real(-inf, inf, name='y')
])

def myobj(point: dict):

    def ackley(x, a=20, b=0.2, c=2*pi ):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        n = len(x)
        s1 = sum( x**2 )
        s2 = sum(cos( c * x ))
        return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)

    x = np.array([point[f'x_{i}'] for i in range(len(point))])
    objective = ackley(x)

    return objective

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )