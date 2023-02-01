import os
import sys
import subprocess
import random
from pathlib import Path
from plopper_cmds import COMMON_FLAGS, CONV_CLANG_CMD, CLANG_CUDA_CMD, XL_CUDA_CMD, CMD2

COMMON_FLAGS.format(utilities_dir, kernel_dir, interimfile, utilities_dir, tmpbinary)

CLANG_CUDA_CMD.format(
    system, gpuarch, COMMON_FLAGS, CUDA_FLAGS
)  # if compiler == "clang" and cuda: else:
CLANG_CUDA_CMD.format(system, gpuarch, COMMON_FLAGS, "")

XL_CUDA_CMD.format(
    system, gpuarch, COMMON_FLAGS, XL_CUDA_FLAGS
)  # if compiler == "clang" and cuda: else:
XL_CUDA_CMD.format(system, gpuarch, COMMON_FLAGS, "")

if system == "pascal":  # LLNL machine or Utah machine
    gpuarch = "sm_60"
elif system == "lassen":  # LLNL machine
    gpuarch = "sm_70"

# CMD2.format(system, utilities_dir, tmpbinary)
CMD2.format(utilities_dir, tmpbinary)

CLANG_CUDA_NVPT_CMD.format(COMMON_FLAGS)

class BasePlopper:
    def __init__(self, sourcefile, outputdir, compiler: str = None):

        self.sourcefile = Path(sourcefile)
        self.outputdir = outputdir + "/tmp_files"
        self.sourcefile_type = self.sourcefile.suffix
        self.compiler = compiler  # "gcc or mpicc"

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return dictVal

    def p2check(self, inputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            for (
                line
            ) in buf:
                if "#P2" in line:
                    return True
        return False

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

    def plotValues_conv(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

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

    def plotValues_conv2d0(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            param = ""  # string to hold the parameters in case we cuda is used
            global cuda
            cuda = False
            for (
                line
            ) in buf:  # check if we are using cuda. If yes, collect the parameters.
                if (
                    "POLYBENCH_2D_ARRAY_DECL_CUDA"
                    or "POLYBENCH_3D_ARRAY_DECL_CUDA"
                    or "POLYBENCH_1D_ARRAY_DECL_CUDA" in line
                ):
                    cuda = True

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

    def findRuntime(self, x, params):
        interimfile = ""
        exetime = 1
        counter = random.randint(
            1, 10001
        )
        interimfile = self.outputdir + "/tmp_" + str(counter) + self.sourcefile_type
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)
        tmpbinary = interimfile if not self.is_compiling else interimfile[:-2]
        kernel_idx = self.sourcefile.rfind("/")
        kernel_dir = self.sourcefile[:kernel_idx]
        utilities_dir = kernel_dir + "/utilities"

        if self.compiler:
            dmpi = "-DMPI" if self.compiler == "mpicc" else ""
            gcc_cmd = MAT_GCC_CMD.format(
                self.compiler,
                dmpi,
                tmpbinary,
                interimfile,
                kernel_dir,
                kernel_dir,
                kernel_dir,
            )
            CONV_CLANG_CMD.format(
                utilities_dir, kernel_dir, interimfile, utilities_dir, tmpbinary
            )
            GPP_BLOCK_CMD.format(kernel_dir, dictval["BLOCK_SIZE"], tmpbinary)
            compilation_status = subprocess.run(
                gcc_cmd, shell=True, stderr=subprocess.PIPE
            ).returncode
        else:
            compilation_status = 0

        run_cmd = kernel_dir + "/exe.pl " + tmpbinary

        if compilation_status == 0:
            execution_status = subprocess.run(
                run_cmd, shell=True, stdout=subprocess.PIPE
            )
            exetime = float(execution_status.stdout.decode("utf-8"))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime  # return execution time as cost

    def findRuntime_conv2d0(self, x, params):
        interimfile = ""
        exetime = 1
        counter = random.randint(
            1, 10001
        )

        interimfile = self.outputdir + "/" + str(counter) + ".c"

        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        tmpbinary = interimfile[:-2]

        kernel_idx = self.sourcefile.rfind("/")
        kernel_dir = self.sourcefile[:kernel_idx]
        kernel_dir = os.path.abspath(kernel_dir)
        utilities_dir = kernel_dir + "/utilities"

        compilation_status = subprocess.run(cmd1, shell=True, stderr=subprocess.PIPE)

        if compilation_status.returncode == 0:

            x = 0
            arr = []
            while x < 2:
                x = x + 1
                execution_status = subprocess.run(
                    cmd2, shell=True, stdout=subprocess.PIPE
                )
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

    def findRuntime_block(self, x, params):
        interimfile = ""
        exetime = 1

        dictVal = self.createDict(x, params)

        tmpbinary = self.outputdir + "/tmp_" + str(uuid.uuid4()) + ".bin"
        kernel_idx = self.sourcefile.rfind("/")
        kernel_dir = self.sourcefile[:kernel_idx]
        gcc_cmd = "g++ " + kernel_dir + "/mmm_block.cpp "
        gcc_cmd += " -D{0}={1}".format("BLOCK_SIZE", dictVal["BLOCK_SIZE"])
        gcc_cmd += " -o " + tmpbinary
        run_cmd = kernel_dir + "/exe.pl " + tmpbinary

        compilation_status = subprocess.run(gcc_cmd, shell=True, stderr=subprocess.PIPE)

        if compilation_status.returncode == 0:
            execution_status = subprocess.run(
                run_cmd, shell=True, stdout=subprocess.PIPE
            )
            exetime = float(execution_status.stdout.decode("utf-8"))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime
