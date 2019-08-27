import os
from autotune import TuningProblem
from autotune.space import *

# system
nodes = 1
cores = 64

# problem space
task_space = Space([
    Integer(100, 100, name='m'),
    Integer(100, 100, name='n'),
])

input_space = Space([
    Integer(1, 10, name='ranks_per_node'),
    Integer(1, 10, name='mb'),
    Integer(1, 10, name='nb'),
    Integer(nodes, nodes*cores, name='nproc'),
    Integer(1, nodes*cores, name='p'),
])

output_space = Space([
    Real(0, inf, name='time')
])

def myobj(point: dict):
    # default params
    params = {
        'fac': 'QR',
        'm': 1000,
        'n': 1000,
        'mb': 32,
        'nb': 32,
        'p': 1,
        'q': 1,
        'tresh': 1.
    }

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

    params.update(point)

    write_input(params)
    os.system('/projects/datascience/regele/ztune/examples/scalapack-driver/bin/theta/pdqrdriver 2>> QR.err')

    def read_output():

        mytime = inf

        with open("QR.out", 'r') as fout:
            for line in fout:
                words = line.split()
                # WRITE( NOUT, FMT = 9993 ) 'WALL', M, N, MB, NB, NPROW, NPCOL, WTIME( 1 ), TMFLOPS, PASSED, FRESID
                if len(words) > 0 and words[0] == "WALL":
                    if words[9] == "PASSED":
                        tmp_time = float(words[7])
                        if tmp_time < mytime:
                            mytime = tmp_time
                            break

        return mytime
    objective = read_output()

    return objective()

TuningProblem(
    task_space=task_space,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )