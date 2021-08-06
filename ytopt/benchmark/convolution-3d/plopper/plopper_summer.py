import os
import sys
import subprocess
import random
import re

class Plopper:
    def __init__(self,sourcefile,outputdir):

        # Initilizing global variables
        self.sourcefile = sourcefile
        self.outputdir = outputdir+"/tmp_files"

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    #Creating a dictionary using parameter label and value
    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return(dictVal)

    def p2check(self, inputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            for line in buf: #check if we are using cuda. If yes, collect the parameters.
                if "#P2" in line:
                    return True
        return False 

    #Replace the Markers in the source file with the corresponding Pragma values
    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
            param = "" #string to hold the parameters in case we cuda is used
            global cuda
            cuda = False
            for line in buf: #check if we are using cuda. If yes, collect the parameters.
                if "POLYBENCH_2D_ARRAY_DECL_CUDA" or "POLYBENCH_3D_ARRAY_DECL_CUDA" or "POLYBENCH_1D_ARRAY_DECL_CUDA"in line:
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
                            modify_line = re.sub(r"#P"+m.group(1), dictVal["P"+m.group(1)], modify_line)
                except Exception as e:
                    print("we got exception", e)
                    print(dictVal)
                    sys.exit(1)
                if modify_line != line:
                    f2.write(modify_line)
                else:
                    #To avoid writing the Marker
                    f2.write(line)

    # Function to find the execution time of the interim file, and return the execution time as cost to the search module
    def findRuntime(self, x, params):
        interimfile = ""
        #exetime = float('inf')
        #exetime = sys.maxsize
        exetime = 1
        counter = random.randint(1, 10001) # To reduce collision increasing the sampling intervals

        interimfile = self.outputdir+"/"+str(counter)+".c"

        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        #compile and find the execution time
        tmpbinary = interimfile[:-2]

        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]
        kernel_dir = os.path.abspath(kernel_dir)
        utilities_dir = kernel_dir+"/utilities"

        commonflags = f"""-DEXTRALARGE_DATASET -DPOLYBENCH_TIME -I{utilities_dir} -I{kernel_dir} {interimfile} {utilities_dir}/polybench.c -o {tmpbinary} -lm -g """

        #with the new script
    
        system = os.environ.get("host", "lassen")
        compiler = os.environ.get("compiler", "xl")
        print("running plopper", system, compiler)
                
        if system == "pascal" : #LLNL machine or Utah machine
            gpuarch = "sm_60"
        elif system == "lassen" : #LLNL machine
            gpuarch = "sm_70"
        else :
            raise RuntimeError("gpuarch unknown, host was not found.")
        
        if compiler == "clang" and cuda:
            cmd1 = f"""invoke_test {system} clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march={gpuarch} {commonflags} -I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64 -Wl,-rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread"""
        elif compiler == "clang" and not cuda:
            cmd1 = f"""invoke_test {system} clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march={gpuarch} {commonflags}"""

        elif compiler == "xl" and cuda: #not sure that this command is correct
            cmd1 = f"""invoke_test {system} /usr/tce/packages/xl/xl-2021.03.11/bin/xlc++ -qsmp -qoffload -qtgtarch={gpuarch} {commonflags} -I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64 -Wl,-rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread """
        elif compiler == "xl" and not cuda:
            cmd1 = f"""invoke_test {system} /usr/tce/packages/xl/xl-2021.03.11/bin/xlc++ -qsmp -qoffload -qtgtarch={gpuarch} {commonflags}"""
#-std=c++11
        else :
            raise RuntimeError("compiler error")

        #with the new script
        cmd2 = f"""invoke_test {system} {utilities_dir}/time_benchmark.sh {tmpbinary}"""
        
            
        print(cmd1)
        print(cmd2)
      
        #Find the compilation status using subprocess
        compilation_status = subprocess.run(cmd1, shell=True, stderr=subprocess.PIPE)

        #Find the execution time only when the compilation return code is zero, else return infinity
        if compilation_status.returncode == 0 :
        #and len(compilation_status.stderr) == 0: #Second condition is to check for warnings
            #loop here to get multiple runs and check if they vary, if do loop here, modify time_benchmark to 2 and NOT 5
            x = 0
            arr = []
            #exetime = 100000
            while(x < 2):
                x = x+1
                execution_status = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE)
            #exetime = float(execution_status.stdout.decode('utf-8'))
                with open(f"{tmpbinary}.out") as cmd2out:
                    #if exetime < float(cmd2out.read()):
                    exetime = float(cmd2out.read())
                    print("hello we are here", exetime)
                    arr.append(exetime)
                if exetime == 0:
                    exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")

        #return exetime #return execution time as cost
        return min(arr)
