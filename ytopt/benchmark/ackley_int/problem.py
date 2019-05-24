import os

import numpy as np

from ytopt.problem import Problem

HERE = os.path.dirname(os.path.abspath(__file__))

template = ""
nparam = 10
for i in range(1, nparam+1):
    template += f" --p{i} {'{}'}"

Problem = Problem(
    app_name='Ackley',
    app_exe=f"python {os.path.join(HERE, 'executable.py')}",
    args_template=template
)

Problem.resources['threads_per_rank'] = [1, 2, 3]

a, b = -15, 30
for i in range(nparam):
    Problem.spec_dim(p_id=i, p_space=[a+i for i in range(b-a)], default=a)
Problem.checkcfg()

if __name__ == '__main__':
    print(Problem)
