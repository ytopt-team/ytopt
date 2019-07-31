import os
from autotune import TuningProblem
from autotune.space import *

task_space = Space([])

input_space = Space([
    Integer(1, 10, name='ranks_per_node'),
    Integer(1, 10, name='mb'),
    Integer(1, 10, name='nb')
])

def myobj(point):

    def write_input(params):
        with open("QR.in", "w") as f:
            f.write("%d\n"%(1))
            f.write("%s, %d, %d, %d, %d, %d, %d, %f\n"%(
                params['fac'],
                params['m'],
                params['n'],
                params['mb'],
                params['nb'],
                params['p'],
                params['q'],
                params['thresh']))

    write_input(point)
    os.system('/projects/datascience/regele/ztune/examples/scalapack-driver/bin/theta/pdqrdriver 2>> QR.err')
    return 0

TuningProblem(
    task_space=task_space,
    input_space=input_space,
    objective=myobj,
    constraints=[],
    model=None
    )