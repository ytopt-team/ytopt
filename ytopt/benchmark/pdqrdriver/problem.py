import os
from ytopt.problem import Problem

HERE = os.path.dirname(os.path.abspath(__file__))

Problem = Problem(
    app_name='pdqrdriver',
    app_exe=f"python {os.path.join(HERE, 'executable.py')}",
    args_template="--m 1000 --n 1000 --mb {} --nb {} --p {}"
)

Problem.spec_dim(p_id=0, p_space=(1, 1000), default=1) # mb
Problem.spec_dim(p_id=1, p_space=(1, 1000), default=1) # nb


Problem.resources['threads_per_rank'] = (1, 64)
Problem.resources['threads_per_core'] = (1, 4)
Problem.resources['cpu_affinity'] = ['none', 'depth']
Problem.resources['ranks_per_node'] = (1, 64)
Problem.resources['num_nodes'] = 1

num_cores_per_node = 64
Problem.spec_dim(
    p_id=2,
    p_space=(1, Problem.resources['num_nodes']*num_cores_per_node),
    default=1) # p

Problem.resources['env'] = 'OMP_NUM_THREADS=64'

Problem.checkcfg()

if __name__ == '__main__':
    print(Problem)
