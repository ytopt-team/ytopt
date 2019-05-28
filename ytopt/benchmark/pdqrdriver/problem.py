import os
from ytopt.problem import Problem

HERE = os.path.dirname(os.path.abspath(__file__))

Problem = Problem(
    app_name='pdqrdriver',
    app_exe=f"python {os.path.join(HERE, 'executable.py')}",
    args_template=""
)

Problem.resources['threads_per_rank'] = [1, 2, 3]

Problem.checkcfg()

if __name__ == '__main__':
    print(Problem)
