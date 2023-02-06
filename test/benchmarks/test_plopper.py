import os
from pathlib import Path
from ytopt.benchmark.plopper.plublic_plopper import BasePlopper, PyPlopper, CompilePlopper

this = Path(__file__)

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

def test_base_plot():
    pass

def test_compiler_init():
    pass

def test_compiler_runtime():
    pass

def test_compiler_runtime_block():
    pass

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
    test_base_plot()
    test_compiler_init()
    test_compiler_runtime()
    test_compiler_runtime_block()
    test_py_init()
    test_py_p2check()
    test_py_checkcuda()
    test_py_plot()
    test_py_runtime()