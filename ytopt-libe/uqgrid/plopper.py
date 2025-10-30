import os
import sys
import subprocess
import random
import psutil

class Plopper:
    def __init__(self,sourcefile,outputdir):

        # Initilizing global variables
        self.sourcefile = sourcefile
        self.outputdir = outputdir

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    #Creating a dictionary using parameter label and value
    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return(dictVal)

    #Replace the Markers in the source file with the corresponding values
    def plotValues(self, dictVal, inputfile, outputfile):
        with open(inputfile, "r") as f1:
            buf = f1.readlines()

        with open(outputfile, "w") as f2:
            for line in buf:
                modify_line = line
                for key, value in dictVal.items():
                    if key in modify_line:
                        if value != 'None': #For empty string options
                            if  len(dictVal) > 10: # for more than 10 tunable parameters
                                modify_line = modify_line.replace('#'+key+' ', str(value))
                            else:
                                modify_line = modify_line.replace('#'+key, str(value))

                if modify_line != line:
                    f2.write(modify_line)
                else:
                    #To avoid writing the Marker
                    f2.write(line)

    # Function to find the execution time of the interim file, and return the execution time as cost to the search module
    def findRuntime(self, x, params, worker):
        interimfile = ""
        #exetime = float('inf')
        exetime = sys.maxsize
        counter = random.randint(1, 10001) # To reduce collision increasing the sampling intervals

        tmpconfig = self.outputdir+"/"+str(counter)+".yaml"

        # Generate intermediate file
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, tmpconfig)

        # create gen and tsi executables
        #tmpgen = "../../generate_scenarios.py" + " --config " + tmpconfig
        tmpgen = "../../generate_scenarios.py" + " --config " + "../"+str(counter)+".yaml"
        #tmpgen = "../../generate_scenarios.py" 
        #tmptsi = "../../TSI_analysis.py" 
        tmptsi = "../../TSI_analysis_parallel.py" 

        cmdgen = "../../exe.pl " + tmpgen 
        cmdtsi = "../../exe.pl " + tmptsi

        # Create several subfolers
        cdir = self.outputdir
        nfolders = 2
        for ncount in range(1, nfolders+1):
            folder = cdir + "/folder_" + str(ncount)
            if not os.path.exists(folder):
                os.makedirs(folder)

            # execute gen and tsi

            # Change the current working directory to folder
            try:
                    os.chdir(folder)
                    print(f"Successfully changed directory to: {os.getcwd()}")
            except FileNotFoundError:
                    print(f"Error: The directory '{folder}' was not found.")

            execution_status = subprocess.Popen(cmdgen, shell=True, stdout=subprocess.PIPE)
            app_timeout = 5000
            try:
                    outs, errs = execution_status.communicate(timeout=app_timeout)
            except subprocess.TimeoutExpired:
                    execution_status.kill()
                    outs, errs = execution_status.communicate()
                    return app_timeout

            execution_status1 = subprocess.Popen(cmdtsi, shell=True, stdout=subprocess.PIPE)
            try:
                    outs1, errs1 = execution_status1.communicate(timeout=app_timeout)
            except subprocess.TimeoutExpired:
                    execution_status1.kill()
                    outs1, errs1 = execution_status1.communicate()
                    return app_timeout

            # change back to the main folder
            folder1 = "../.." 
            try:
                    os.chdir(folder1)
                    print(f"Successfully changed directory to: {os.getcwd()}")
            except FileNotFoundError:
                    print(f"Error: The directory '{folder1}' was not found.")

        exetime = outs1.strip()

        return exetime #return execution time as cost

