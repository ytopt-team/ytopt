import os
import sys
import subprocess
import random
import psutil

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
    def findRuntime(self, x, params, worker):
        interimfile = ""
        #exetime = sys.maxsize
        counter = random.randint(1, 10001) # To reduce collision increasing the sampling intervals
        interimfile = self.outputdir+"/"+str(counter)+".C"
        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)

        #compile and find the execution time
        tmpbinary = interimfile[:-2] + '_w' + str(worker)

        kernel_idx = self.sourcefile.rfind('/')
        kernel_dir = self.sourcefile[:kernel_idx]
        
        cmd1 = "ftn -O3 -fopenmp -c ./src/type_defs.f90 " \
                + " ; CC -O3 -I./src -fopenmp -DSW4_OPENMP -DSW4_CROUTINES -I./src/double "  + \
                " -o " + tmpbinary + " ./src/main.C " + interimfile +" " + \
                "./src/Sarray.C ./src/Source.C ./src/SuperGrid.C ./src/GridPointSource.C ./src/time_functions.C ./src/EW_cuda.C ./src/ew-cfromfort.C ./src/EWCuda.C ./src/CheckPoint.C ./src/Parallel_IO.C ./src/EW-dg.C ./src/MaterialData.C ./src/MaterialBlock.C ./src/Polynomial.C ./src/SecondOrderSection.C ./src/Filter.C ./src/TimeSeries.C ./src/sacsubc.C ./src/curvilinear-c.C " \
                + " -I" + kernel_dir + \
                " -lm" 
                
        cmd2 = kernel_dir + "/exe.pl " +  tmpbinary
        
        #Find the compilation status using subprocess
        compilation_status = subprocess.run(cmd1, shell=True, stderr=subprocess.PIPE)

        #Find the execution time only when the compilation return code is zero, else return infinity
        if compilation_status.returncode == 0 :
        #and len(compilation_status.stderr) == 0: #Second condition is to check for warnings
           execution_status = subprocess.Popen(cmd2, shell=True, stdout=subprocess.PIPE)
           app_timeout = 500

           try:
                outs, errs = execution_status.communicate(timeout=app_timeout)
           except subprocess.TimeoutExpired:
                execution_status.kill()
                for proc in psutil.process_iter(attrs=['pid', 'name']):
                    if 'exe.pl' in proc.info['name']:
                        proc.kill()
                outs, errs = execution_status.communicate()
                return app_timeout

           exetime = float(outs.strip())
        else:
            print(compilation_status.stderr)
            print("compile failed")
        return exetime #return execution time as cost
