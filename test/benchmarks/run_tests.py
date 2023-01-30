import os
import pytest
import subprocess
from pathlib import Path

class Changer:

    def __init__(self, testdir, home=os.getcwd()):
        self.testdir = testdir
        self.home = home

    def __enter__(self):
        os.chdir(self.testdir)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.home)

def run_test(testdir):
    with Changer(testdir):
        assert not subprocess.check_call(["./run.bat"])

def test_dl():
    run_test(Path("dl") / "mnist")

def test_xsbench_mpi_omp():
    run_test(Path("xsbench-mpi-omp") / "xsbench")

def test_xsbench_omp():
    run_test(Path("xsbench-omp") / "xsbench")

if __name__ == "__main__":
    test_dl()
    test_xsbench_mpi_omp()
    test_xsbench_omp()