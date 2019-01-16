from collections import OrderedDict
import numpy as np

# class Problem():
#     def __init__(self):
#         nparam = 10
#         space = OrderedDict()
#         #problem specific parameters
#         a, b = -15, 30
#         for i in range(nparam):
#             space['p%d'%(i+1)] = [a+i for i in range(b-a)]
#         self.space = space
#         self.params = self.space.keys()
#         self.starting_point = [-15] * nparam

from ytopt.problem import Problem

cmd_frmt = "python /Users/romainegele/Documents/Argonne/ytopt/problems/ackley_int/executable.py"
nparam = 10
for i in range(1, nparam+1):
    cmd_frmt += f" --p{i} {'{}'}"
problem = Problem(cmd_frmt)

a, b = -15, 30
for i in range(nparam):
    problem.spec_dim(p_id=i, p_space=[a+i for i in range(b-a)], default=-15)
problem.checkcfg()

if __name__ == '__main__':
    print(pb)
