from collections import OrderedDict
import numpy as np
import os 


HERE = os.path.dirname(os.path.abspath(__file__))

from ytopt.problem import Problem

cmd_frmt = "python " +HERE+"/executable.py"
nparam = 10
for i in range(1, nparam+1):
    cmd_frmt += f" --p{i} {'{}'}"
problem = Problem(cmd_frmt)

a, b = -5, 10
for i in range(nparam):
    problem.spec_dim(p_id=i, p_space=[i for i in range(a,b+1,1)], default=a)
problem.checkcfg()

if __name__ == '__main__':
    print(pb)
