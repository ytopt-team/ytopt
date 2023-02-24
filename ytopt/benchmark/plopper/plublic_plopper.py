import os
import re
import sys
import subprocess
import random
from pathlib import Path
from ytopt.benchmark.plopper.cmds import (
    COMMON_FLAGS,
    CUDA_FLAGS,
)


class BasePlopper:
    """
    Utility class for preparing input template applications for parameterization in
    separate directories, compiling, then averaging their runtimes over several executions.
    Use the `CompilePlopper` and `PyPlopper` subclasses for experiments that parameterize either
    compiled applications or Python files respectively.
    """

    def __init__(self, sourcefile: Path, outputdir: Path) -> None:

        self.sourcefile = Path(sourcefile)
        self.outputdir = Path(outputdir) / "tmp_files"
        self.sourcefile_type = self.sourcefile.suffix
        self.kernel_dir = self.sourcefile.parent
        self.utilities_dir = self.kernel_dir / "utilities"

        assert self.sourcefile.is_file(), "Unable to find or access sourcefile"

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    def _createDict(self, x, params) -> dict:
        """sets x's as values to params in a new dictionary"""
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return dictVal

    def _get_interimfile(self) -> Path:
        return self.outputdir / (
            "tmp_" + str(random.randint(1, 10001)) + self.sourcefile_type
        )

    def _plotValues(self, dictVal: dict, inputfile: Path, outputfile: Path) -> None:
        """Parameterizes an inputfile using a given dictionary and an outputfile path"""
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value != "None":  # For empty string options
                            modify_line = modify_line.replace("#" + key, str(value))

                if modify_line != line:
                    f2.write(modify_line)
                else:
                    f2.write(line)


class CompilePlopper(BasePlopper):
    """Utility class for preparing, parameterizing, compiling, then averaging runtimes
    for compiled applications. Note that besides instantiating a `obj = CompilePlopper(...)` instance,
    `obj.set_compile_command(command)` should first be called with the string-to-format to compile
    the target application.

    At a minimum `CompilePlopper` requires `sourcefile` and `outputdir` paths, but other key-word parameters
    like `compiler`, `do_cuda`, `system`, and `gpuarch` can be provided to parameterize like-named fields
    in provided compile-commands.
    """

    def __init__(
        self,
        sourcefile: Path,
        outputdir: Path,
        compiler: str = "mpicc",
        do_cuda: bool = False,
        system: str = "local",
        gpuarch: str = None,
    ) -> None:
        self.compiler = compiler
        assert compiler in [
            "clang",
            "xl",
            "mpicc",
            "gcc",
        ], "specify compiler as one of clang, xl, mpicc, or gcc"
        self.do_cuda = do_cuda
        self.system = system
        self.gpuarch = gpuarch
        super().__init__(sourcefile, outputdir)

    def set_compile_command(self, command: str) -> None:
        """Provide a string with {fields} to parameterize and form a command
        to compile interim binaries of a target application.
        """
        self.compile_cmd = command

    def _plotValues(self, dictVal, inputfile, outputfile):
        """Parameterizes an inputfile using a given dictionary and an outputfile path"""
        buf = self._check_cuda(inputfile)

        with open(outputfile, "w") as f2:
            for line in buf:
                stop = False
                modify_line = line
                try:
                    while not stop:
                        if not re.search(r"#P([0-9]+)", modify_line):
                            stop = True
                        for m in re.finditer(r"#P([0-9]+)", modify_line):
                            modify_line = re.sub(
                                r"#P" + m.group(1),
                                dictVal["P" + m.group(1)],
                                modify_line,
                            )
                except Exception as e:
                    print("we got exception", e)
                    print(dictVal)
                    sys.exit(1)
                if modify_line != line:
                    f2.write(modify_line)
                else:
                    # To avoid writing the Marker
                    f2.write(line)

    def _run_compile(self, dictVal, interimfile, tmpbinary) -> int:
        """Parameterizes and runs the provided compile-command"""
        dmpi = "-DMPI" if self.compiler == "mpicc" else ""
        options = {
            "compiler": self.compiler,
            "mpi_macro": dmpi,
            "tmpbinary": tmpbinary,
            "interimfile": interimfile,
            "kernel_dir": self.kernel_dir,
            "utilities_dir": self.utilities_dir,
            "system": self.system,
            "block_size": dictVal.get("BLOCK_SIZE", 0),
            "gpuarch": self.gpuarch,
            "cuda_flags": CUDA_FLAGS,
            "CONDA_PREFIX": os.environ.get("CONDA_PREFIX"),
        }
        common_flags = COMMON_FLAGS.format(**options)
        options["common_flags"] = common_flags
        fmt_compile_cmd = self.compile_cmd.format(**options)
        compilerun = subprocess.run(fmt_compile_cmd, shell=True, stderr=subprocess.PIPE)
        return compilerun.returncode

    def findRuntime(self, x, params) -> int:
        dictVal = self._createDict(x, params)
        interimfile = self._get_interimfile()
        self._plotValues(dictVal, self.sourcefile, interimfile)
        tmpbinary = str(interimfile)[:-2]

        compilation_status = self._run_compile(dictVal, interimfile, tmpbinary)
        run_cmd = str(self.kernel_dir) + "/exe.pl " + tmpbinary

        if compilation_status == 0:
            execution_status = subprocess.run(
                run_cmd, shell=True, stdout=subprocess.PIPE
            )
            exetime = float(execution_status.stdout.decode("utf-8"))
            if exetime == 0:
                exetime = 1
        else:
            print("compile failed. errcode: ", compilation_status)
            exetime = 1
        return exetime  # return execution time as cost

    def p2check(self, inputfile: Path) -> bool:
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            for line in buf:
                if "#P2" in line:
                    return True
        return False

    def _check_cuda(self, inputfile: Path):
        self.cuda = False
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            for (
                line
            ) in buf:  # check if we are using cuda. If yes, collect the parameters.
                if (
                    "POLYBENCH_2D_ARRAY_DECL_CUDA"
                    or "POLYBENCH_3D_ARRAY_DECL_CUDA"
                    or "POLYBENCH_1D_ARRAY_DECL_CUDA" in line
                ):
                    self.cuda = True
        return buf


class PyPlopper(BasePlopper):
    def __init__(self, sourcefile: Path, outputdir: Path) -> None:
        super().__init__(sourcefile, outputdir)

    # Replace the Markers in the source file with the corresponding values
    def _plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value != "None":  # For empty string options
                            modify_line = modify_line.replace("#" + key, str(value))

                if modify_line != line:
                    f2.write(modify_line)
                else:
                    # To avoid writing the Marker
                    f2.write(line)

    def findRuntime(self, x, params) -> int:
        exetime = 1
        dictVal = self._createDict(x, params)
        interimfile = self._get_interimfile()
        self._plotValues(dictVal, self.sourcefile, interimfile)
        tmpbinary = interimfile

        cmd2 = str(self.kernel_dir) + "/exe.pl " + str(tmpbinary)

        # Find the execution time
        execution_status = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE)
        exetime = float(execution_status.stdout.decode("utf-8"))
        if exetime == 0:
            exetime = 1
        return exetime
