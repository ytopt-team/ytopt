import os
from unittest import mock
from pathlib import Path
from ytopt.benchmark.plopper import BasePlopper, PyPlopper, CompilePlopper
from ytopt.benchmark.plopper.cmds import *

this = Path(__file__)
srcfile = Path(this.parent / "xsbench-mpi-omp/xsbench/mmp.c")
pysrcfile = Path(this.parent / "dl/mnist/dlp.py")


def test_base_init():
    print("test_base_init")
    obj = BasePlopper(this, this.parent)
    assert obj.sourcefile == this, "Sourcefile not passed through to plopper"
    assert obj.outputdir == this.parent / "tmp_files", "Outputdir not set to plopper"
    assert os.path.exists(
        this.parent
    ), "plopper did not create outputdir, or respect existance"
    assert (
        obj.kernel_dir == this.parent
    ), "plopper did not detect parent dir of sourcefile"
    assert (
        obj.sourcefile_type == ".py"
    ), "plopper did not detect file type of sourcefile"


def test_get_interimfile():
    print("test_get_interimfile")
    obj = BasePlopper(this, this.parent)
    ifile = obj._get_interimfile()
    assert isinstance(ifile, Path), "plopper didn't return path to a file"
    assert ifile.suffix == ".py", "plopper didn't return path to a python file"
    assert "tmp_files" in str(ifile), "interimfile isn't within tmp_files"


def test_compiler_init():
    print("test_compiler_init")
    obj = CompilePlopper(srcfile, this.parent)
    assert obj.compiler == "mpicc", "default compiler not set"
    flag = 1
    try:
        obj = CompilePlopper(srcfile, this.parent, "nocompile")
        flag = 0
    except AssertionError:
        flag = 1
    assert flag == 1, "Plopper didn't error on accepting invalid compiler"
    assert obj.sourcefile == srcfile, "CompilePlopper didn't init superclass"


def test_compiler_runtime():
    print("test_compiler_runtime")
    obj = CompilePlopper(srcfile, this.parent)
    obj.set_compile_command(MAT_GCC_CMD)
    assert obj.compile_cmd == MAT_GCC_CMD, "Compile command not set to plopper"
    # mock compilation and benchmark outputs to test plopper code without actually running an app
    with mock.patch("ytopt.benchmark.plopper.plublic_plopper.subprocess") as apprun:
        apprun.run.return_value.returncode = 0  # mock compiling succeeding
        apprun.run.return_value.stdout = (
            b"3.1"  # mock app running for average 3.1 seconds
        )
        exetime = obj.findRuntime(["4", "64", " "], ["P0", "P1", "P2"])
        assert exetime == 3.1, "exetime wasn't found as non-default value"
        apprun.run.return_value.returncode = "fake error"  # mock compiling failing
        exetime = obj.findRuntime(["4", "64", " "], ["P0", "P1", "P2"])
        assert exetime == 1, "exetime wasn't set to default value after failed compile"
    assert os.path.exists(this.parent / "tmp_files"), "tmp_files directory not created"
    assert len(
        os.listdir(this.parent / "tmp_files")
    ), "parameterized file not placed into tmp_files"


def test_py_init():
    print("test_py_init")
    obj = PyPlopper(pysrcfile, this.parent)
    assert (
        obj.sourcefile_type == ".py"
    ), "plopper did not detect file type of sourcefile"  # maybe redundant


def test_py_runtime():
    print("test_py_runtime")
    obj = PyPlopper(pysrcfile, this.parent)
    with mock.patch("ytopt.benchmark.plopper.plublic_plopper.subprocess") as apprun:
        apprun.run.return_value.stdout = (
            b"6.1"  # mock app running for average 3.1 seconds
        )
        exetime = obj.findRuntime(
            ["100", "24", "0.2", "rmsprop"], ["P1", "P2", "P3", "P4"]
        )
        assert exetime == 6.1, "exetime wasn't found as non-default value"
        apprun.run.return_value.stdout = b"0"  # mock app fail
        exetime = obj.findRuntime(
            ["100", "24", "0.2", "rmsprop"], ["P1", "P2", "P3", "P4"]
        )
        assert exetime == 1, "exetime wasn't set to default value after failed run"


if __name__ == "__main__":
    test_base_init()
    test_get_interimfile()
    test_compiler_init()
    test_compiler_runtime()
    test_py_init()
    test_py_runtime()
    print("Done!")
