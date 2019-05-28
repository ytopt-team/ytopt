import os
from ytopt.problem import Problem

HERE = os.path.dirname(os.path.abspath(__file__))

Problem = Problem(
    app_name='pdqrdriver',
    app_exe=f"python {os.path.join(HERE, 'executable.py')}",
    args_template=""
)

Problem.resources['threads_per_rank'] = 64
Problem.resources['threads_per_core'] = 1
Problem.resources['cpu_binding'] = 'depth'
Problem.resources['ranks_per_node'] = (1, 10)
Problem.resources['num_nodes'] = 1
Problem.resources['env'] = 'OMP_PLACES=threads OMP_PROC_BIND=spread export OMP_NUM_THREADS=64'

Problem.checkcfg()

if __name__ == '__main__':
    print(Problem)
