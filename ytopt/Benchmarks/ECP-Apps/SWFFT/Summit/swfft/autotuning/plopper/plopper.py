import os
import sys
import subprocess
import random

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

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value != 'None': #For empty string options
                            modify_line = modify_line.replace('#'+key, str(value))

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

        interimfile = self.outputdir+"/"+str(counter)+".cpp"


        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        #compile and find the execution time
        tmpbinary = interimfile[:-4]

        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]

        cmd1 =  "mpic++ -O3 -fopenmp -Wall -DDFFT_TIMING=0 -I/opt/cray/pe/fftw/3.3.8.6/mic_knl/include " \
                + " -I" +kernel_dir + " -c -o " + tmpbinary + ".o " + interimfile \
                + " ;  mpicc -O3 -std=gnu99 -fopenmp -Wall -Wno-deprecated -DDFFT_TIMING=0 -I/opt/cray/pe/fftw/3.3.8.6/mic_knl/include " \
                + " -I" +kernel_dir +" -c -o " +kernel_dir + "/distribution.o " + kernel_dir + "/distribution.c " \
                + " ; mpic++ -O3 -fopenmp -Wall "  \
                + " -o " + tmpbinary + " " + tmpbinary + ".o" +" " + kernel_dir + "/distribution.o " \
                + " -L/opt/cray/pe/fftw/3.3.8.6/mic_knl/lib -lfftw3_omp -lfftw3 -lm "

        cmd2 = kernel_dir + "/exe.pl " +  tmpbinary

        #Find the compilation status using subprocess
        compilation_status = subprocess.run(cmd1, shell=True, stderr=subprocess.PIPE)

        #Find the execution time only when the compilation return code is zero, else return infinity
        if compilation_status.returncode == 0 :
        #and len(compilation_status.stderr) == 0: #Second condition is to check for warnings
            execution_status = subprocess.run(cmd2, shell=True, stdout=subprocess.PIPE)
            exetime = float(execution_status.stdout.decode('utf-8'))
            if exetime == 0:
               exetime = 1
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime #return execution time as cost

