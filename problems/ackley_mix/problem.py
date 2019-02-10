from collections import OrderedDict
import numpy as np
import os 

np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))

from ytopt.problem import Problem

cmd_frmt = "python " +HERE+"/executable.py"
nparam = 6

for i in range(0, nparam):
    cmd_frmt += f" --p{i} {'{}'}"
problem = Problem(cmd_frmt)

a, b = -15, 30

problem.spec_dim(p_id=0, p_space=(a, b), default=a)
problem.spec_dim(p_id=1, p_space=(a, b), default=a)
problem.spec_dim(p_id=2, p_space=[a+i for i in range(b-a)], default=a)
problem.spec_dim(p_id=3, p_space=[a+i for i in range(b-a)], default=a)
problem.spec_dim(p_id=4, p_space= list(np.random.permutation([str(a+i) for i in range(b-a)])), default=str(a))
problem.spec_dim(p_id=5, p_space= list(np.random.permutation([str(a+i) for i in range(b-a)])), default=str(a))

problem.checkcfg()

if __name__ == '__main__':
    print(problem)
