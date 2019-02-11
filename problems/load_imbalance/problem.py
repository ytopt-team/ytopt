from collections import OrderedDict
import numpy as np
import os 

np.random.seed(0)

HERE = os.path.dirname(os.path.abspath(__file__))

from ytopt.problem import Problem

cmd_frmt = "python " +HERE+"/executable.py"
nparam = 32
nb_classes = 8
for i in range(nparam):
    cmd_frmt += f" --p{i} {'{}'}"
problem = Problem(cmd_frmt)

for i in range(nparam):
    problem.spec_dim(p_id=i, p_space=[str(i) for i in list(range(nb_classes))], default='0')
problem.checkcfg()

if __name__ == '__main__':
    print(problem)

