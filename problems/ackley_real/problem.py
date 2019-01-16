# from collections import OrderedDict
# class Problem():
#     def __init__(self):
#         nparam = 10
#         space = OrderedDict()
#         #problem specific parameters
#         for i in range(nparam):
#             space['p%d'%(i+1)] = (-15,30)
#         self.space = space
#         self.params = self.space.keys()
#         self.starting_point = [-15] * nparam

from ytopt.problem import Problem

cmd_frmt = "python /Users/romainegele/Documents/Argonne/ytopt/problems/ackley_real/executable.py --p1 {} --p2 {} --p3 {} --p4 {} --p5 {} --p6 {} --p7 {} --p8 {} --p9 {} --p10 {}"
problem = Problem(cmd_frmt)

a, b = -15, 30
nparams = 10
for i in range(nparams):
    problem.spec_dim(p_id=i, p_space=(a, b), default=-15)
problem.checkcfg()

if __name__ == '__main__':
    print(problem)

