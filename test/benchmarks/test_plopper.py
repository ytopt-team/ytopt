import os
from pathlib import Path
from ytopt.benchmark.plopper import BasePlopper, PyPlopper, CompilePlopper
from ytopt.benchmark.plopper.cmds import *

this = Path(__file__)
srcfile = Path(this.parent / "xsbench-mpi-omp/xsbench/mmp.c")

def test_base_init():
    obj = BasePlopper(this, this.parent)
    assert obj.sourcefile == this, "Sourcefile not passed through to plopper"
    assert obj.outputdir == this.parent / "tmp_files", "Outputdir not set to plopper"
    assert os.path.exists(this.parent), "plopper did not create outputdir, or respect existance"
    assert obj.kernel_dir == this.parent, "plopper did not detect parent dir of sourcefile"
    assert obj.sourcefile_type == ".py", "plopper did not detect file type of sourcefile"

def test_get_interimfile():
    obj = BasePlopper(this, this.parent)
    ifile = obj.get_interimfile()
    assert isinstance(ifile, Path), "plopper didn't return path to a file"
    assert ifile.suffix == ".py", "plopper didn't return path to a python file"
    assert "tmp_files" in str(ifile), "interimfile isn't within tmp_files"    

def test_compiler_init():
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
    obj = CompilePlopper(srcfile, this.parent)
    obj.set_compile_command(MAT_GCC_CMD)
    assert obj.compile_cmd == MAT_GCC_CMD, "Compile command not set to plopper"
    exetime = obj.findRuntime([4, '64', ' '],['P0', 'P1', 'P2'])
    assert exetime != 1, "exetime wasn't found as non-default value"

def test_py_init():
    pass

def test_py_p2check():
    pass

def test_py_checkcuda():
    pass

def test_py_plot():
    pass

def test_py_runtime():
    pass

if __name__ == "__main__":
    test_base_init()
    test_get_interimfile()
    test_compiler_init()
    test_compiler_runtime()
    test_py_init()
    test_py_p2check()
    test_py_plot()
    test_py_runtime()