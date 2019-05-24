from collections import OrderedDict
import numpy as np
import os

np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))

from ytopt.problem import Problem


template = ""
nparam = 6

for i in range(0, nparam):
    template += f" --p{i} {'{}'}"

Problem = Problem(
    app_name='Ackley',
    app_exe=f"python {os.path.join(HERE, 'executable.py')}",
    args_template=template
)

a, b = -15, 30

Problem.spec_dim(p_id=0, p_space=(a, b), default=a)
Problem.spec_dim(p_id=1, p_space=(a, b), default=a)
Problem.spec_dim(p_id=2, p_space=[a+i for i in range(b-a)], default=a)
Problem.spec_dim(p_id=3, p_space=[a+i for i in range(b-a)], default=a)
Problem.spec_dim(p_id=4, p_space= list(np.random.permutation([str(a+i) for i in range(b-a)])), default=str(a))
Problem.spec_dim(p_id=5, p_space= list(np.random.permutation([str(a+i) for i in range(b-a)])), default=str(a))

Problem.checkcfg()

if __name__ == '__main__':
    print(Problem)
