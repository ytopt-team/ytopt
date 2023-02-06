import os
import sys
import subprocess
import random
from pathlib import Path
from plopper_cmds import (
    COMMON_FLAGS,
    CONV_CLANG_CMD,
    CLANG_CUDA_CMD,
    XL_CUDA_CMD,
    CMD2,
    CUDA_FLAGS,
    XL_CUDA_FLAGS,
    GPP_BLOCK_CMD,
)

# COMMON_FLAGS.format(utilities_dir, kernel_dir, interimfile, utilities_dir, tmpbinary)

# CLANG_CUDA_CMD.format(
#     system, gpuarch, COMMON_FLAGS, CUDA_FLAGS
# )  # if compiler == "clang" and cuda: else:
# CLANG_CUDA_CMD.format(system, gpuarch, COMMON_FLAGS, "")

# XL_CUDA_CMD.format(
#     system, gpuarch, COMMON_FLAGS, XL_CUDA_FLAGS
# )  # if compiler == "clang" and cuda: else:
# XL_CUDA_CMD.format(system, gpuarch, COMMON_FLAGS, "")

# if system == "pascal":  # LLNL machine or Utah machine
#     gpuarch = "sm_60"
# elif system == "lassen":  # LLNL machine
#     gpuarch = "sm_70"

# # CMD2.format(system, utilities_dir, tmpbinary)
# CMD2.format(utilities_dir, tmpbinary)

# CLANG_CUDA_NVPT_CMD.format(COMMON_FLAGS)


class BasePlopper:
    def __init__(self, sourcefile: Path, outputdir: Path):

        self.sourcefile = Path(sourcefile)
        self.outputdir = outputdir / "tmp_files"
        self.sourcefile_type = self.sourcefile.suffix
        self.kernel_dir = self.sourcefile.parent
        self.utilities_dir = self.kernel_dir / "utilities"

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return dictVal

    def get_interimfile(self):
        return self.outputdir / ("tmp_" + str(random.randint(1, 10001)) + self.sourcefile_type)

    def plotValues(self, dictVal, inputfile, outputfile):
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
    def __init__(self, sourcefile, outputdir, compiler: str = "mpicc", do_cuda: bool = False):
        self.compiler = compiler  # clang or xl or mpicc
        assert compiler in ["clang", "xl", "mpicc", "gcc"], \
             "specify compiler as one of clang, xl, mpicc, or gcc"
        self.do_cuda = do_cuda
        super().__init__(sourcefile, outputdir)

    def set_compile_command(command:str):
        self.compile_cmd = command

    def run_compile():
        pass

    def run_tmpbinary():
        pass

    def findRuntime(self, x, params):
        interimfile = self.get_interimfile()
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)
        tmpbinary = interimfile[:-2]

        dmpi = "-DMPI" if self.compiler == "mpicc" else ""
        gcc_cmd = MAT_GCC_CMD.format(
            self.compiler,
            dmpi,
            tmpbinary,
            interimfile,
            self.kernel_dir,
            self.kernel_dir,
            self.kernel_dir,
        )
        CONV_CLANG_CMD.format(utilities_dir, kernel_dir, interimfile, utilities_dir, tmpbinary)
        GPP_BLOCK_CMD.format(kernel_dir, dictval["BLOCK_SIZE"], tmpbinary)
        compilation_status = subprocess.run(gcc_cmd, shell=True, stderr=subprocess.PIPE).returncode

        run_cmd = self.kernel_dir + "/exe.pl " + tmpbinary

        if compilation_status == 0:
            execution_status = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE)
            exetime = float(execution_status.stdout.decode("utf-8"))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
            exetime = 1
        return exetime  # return execution time as cost

    def findRuntime_block(self, x, params):
        dictVal = self.createDict(x, params)
        tmpbinary = self.outputdir + "/tmp_" + str(uuid.uuid4()) + ".bin"
        gcc_cmd = GPP_BLOCK_CMD.format(kernel_dir, "BLOCK_SIZE", dictVal["BLOCK_SIZE"], tmpbinary)
        run_cmd = self.kernel_dir + "/exe.pl " + tmpbinary

        compilation_status = subprocess.run(gcc_cmd, shell=True, stderr=subprocess.PIPE)

        if compilation_status.returncode == 0:
            execution_status = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE)
            exetime = float(execution_status.stdout.decode("utf-8"))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime


class PyPlopper(BasePlopper):
    def __init__(self, sourcefile, outputdir):
        super().__init__(sourcefile, outputdir)

    def findRuntime(self, x, params):
        exetime = 1
        interimfile = self.get_interimfile()
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)
        tmpbinary = interimfile[:-2]

        compilation_status = subprocess.run(cmd1, shell=True, stderr=subprocess.PIPE)

        if compilation_status.returncode == 0:

            x = 0
            arr = []
            while x < 2:
                x = x + 1
                execution_status = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE)
                with open(f"{tmpbinary}.out") as cmd2out:
                    exetime = float(cmd2out.read())
                    print("hello we are here", exetime)
                    arr.append(exetime)
                if exetime == 0:
                    exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")

        return min(arr)

    def conv_p2check(self, inputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            for line in buf:
                if "#P2" in line:
                    return True
        return False

    def _check_cuda(self, inputfile):
        self.cuda = False
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            for line in buf:  # check if we are using cuda. If yes, collect the parameters.
                if (
                    "POLYBENCH_2D_ARRAY_DECL_CUDA"
                    or "POLYBENCH_3D_ARRAY_DECL_CUDA"
                    or "POLYBENCH_1D_ARRAY_DECL_CUDA" in line
                ):
                    self.cuda = True
        return buf

    def plotValues(self, dictVal, inputfile, outputfile):
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
                    f2.write(line)
