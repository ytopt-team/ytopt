from collections import OrderedDict
import numpy as np
import os 


HERE = os.path.dirname(os.path.abspath(__file__))

from ytopt.problem import Problem

cmd_frmt = "python " +HERE+"/executable.py"
nparam = 5
for i in range(0, nparam):
    cmd_frmt += f" --p{i} {'{}'}"
problem = Problem(cmd_frmt)
     
# example layer vgg2_1
#[N=64,K=128,C=64,H=112,W=112,R=S=3,stride=1,pad=1]
problem.spec_dim(p_id=0, p_space=[1, 2, 4], default=1) # K_B K block
problem.spec_dim(p_id=1, p_space=[1, 2], default=1) # C_B C block 
problem.spec_dim(p_id=2, p_space=[1, 2, 4, 7, 8, 14, 16, 28, 56, 112], default=1) #W_B W block 
problem.spec_dim(p_id=3, p_space=[x for x in range(2, 42, 2)], default=1) # U_W unroll
problem.spec_dim(p_id=4, p_space=[x for x in range(2, 18, 2)], default=1) # U_W unroill
problem.checkcfg()

if __name__ == '__main__':
    print(problem)
