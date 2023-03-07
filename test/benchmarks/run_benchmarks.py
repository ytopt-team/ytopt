import os
import pytest
import importlib
import subprocess
from unittest import mock
from pathlib import Path
from ytopt.search.optimizer import Optimizer
from ytopt.search import Search
from ytopt.search import util
from ytopt.search.ambs import AMBS


class Changer:
    def __init__(self, testdir, home=os.getcwd()):
        self.testdir = testdir
        self.home = home

    def __enter__(self):
        os.chdir(self.testdir)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.home)


def get_problem(path):
    return f"ytopt.benchmark.{str(path).replace('/', '.')}.problem.Problem"


def run_test(testdir, margs):
    with Changer(testdir):
        margs = AMBS.parse_args(margs.split())
        search = AMBS(**vars(margs))
        search.main()


def test_dl():
    print("test_dl", flush=True)
    path = Path("dl") / "mnist"
    args = f"--evaluator ray --problem {get_problem(path)} --max-evals=2 --learner RF"
    run_test(path, args)


def test_xsbench_mpi_omp():
    print("test_xsbench_mpi_omp", flush=True)
    path = Path("xsbench-mpi-omp") / "xsbench"
    args = f"--evaluator ray --problem {get_problem(path)} --max-evals=2 --learner RF"
    run_test(path, args)


def test_xsbench_omp():
    print("test_xsbench_omp", flush=True)
    path = Path("xsbench-omp") / "xsbench"
    args = f"--evaluator ray --problem {get_problem(path)} --max-evals=2 --learner RF"
    run_test(path, args)


if __name__ == "__main__":
    test_dl()
    test_xsbench_mpi_omp()
    test_xsbench_omp()
    print("Done!", flush=True)
