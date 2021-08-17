import os, sys, subprocess, random, re
random.seed(1234)

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

    #Replace the Markers in the source file with the corresponding Pragma values
    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()
#             param = "" #string to hold the parameters in case we cuda is used
#             global cuda
#             cuda = False
#             for line in buf: #check if we are using cuda. If yes, collect the parameters.
#                 if "POLYBENCH_2D_ARRAY_DECL_CUDA" or "POLYBENCH_3D_ARRAY_DECL_CUDA" or "POLYBENCH_1D_ARRAY_DECL_CUDA"in line:
#                     cuda = True

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
        exetime = 1
        counter = random.randint(1, 10001) # To reduce collision increasing the sampling intervals

        interimfile = self.outputdir+"/tmp_"+str(counter)+".c"

        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        #compile and find the execution time
        tmpbinary = interimfile[:-2]

        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]
        utilities_dir = kernel_dir+"/utilities"

        commonflags = f"""-DEXTRALARGE_DATASET -DPOLYBENCH_TIME -I{utilities_dir} -I{kernel_dir} {interimfile} {utilities_dir}/polybench.c -o {tmpbinary} -lm -g """
        
        gcc_cmd = f"""clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75 {commonflags} -I/soft/compilers/cuda/cuda-11.4.0/include -L/soft/compilers/cuda/cuda-11.4.0/lib64 -Wl,-rpath=/soft/compilers/cuda/cuda-11.4.0/lib64 -lcudart_static -ldl -lrt -pthread"""
        
        run_cmd = kernel_dir + "/exe.pl " + tmpbinary
#         print (run_cmd)
        #Find the compilation status using subprocess
        compilation_status = subprocess.run(gcc_cmd, shell=True, stderr=subprocess.PIPE)

        #Find the execution time only when the compilation return code is zero, else return infinity
        if compilation_status.returncode == 0 :
            execution_status = subprocess.run(run_cmd, shell=True, stdout=subprocess.PIPE)
            exetime = float(execution_status.stdout.decode('utf-8'))
            if exetime == 0:
                exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
            
        #return execution time as cost
        return exetime
