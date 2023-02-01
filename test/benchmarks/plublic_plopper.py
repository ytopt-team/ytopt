import os
import sys
import subprocess
import random
from pathlib import Path

class Plopper:
    def __init__(self, sourcefile, outputdir):

        # Initilizing global variables
        self.sourcefile = Path(sourcefile)
        self.outputdir = outputdir + "/tmp_files"
        self.sourcefile_type = self.sourcefile.suffix

        if not os.path.exists(self.outputdir):
            os.makedirs(self.outputdir)

    # Creating a dictionary using parameter label and value
    def createDict(self, x, params):
        dictVal = {}
        for p, v in zip(params, x):
            dictVal[p] = v
        return dictVal


    # Replace the Markers in the source file with the corresponding values
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
                    # To avoid writing the Marker
                    f2.write(line)

    def findRuntime(self, x, params):
        interimfile = ""
        # exetime = float('inf')
        # exetime = sys.maxsize
        exetime = 1
        counter = random.randint(
            1, 10001
        )  # To reduce collision increasing the sampling intervals
        interimfile = self.outputdir + "/tmp_" + str(counter) + self.sourcefile_type
        dictVal = self.createDict(x, params)
        self.plotValues(dictVal, self.sourcefile, interimfile)