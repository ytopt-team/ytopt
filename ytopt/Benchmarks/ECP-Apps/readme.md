This directory includes all files for autotuning four ECP proxy apps: AMG, SW4lite, SWFFT, and XSBench at large scales on ALCF Theta and OLCF Summit, and one simplified autotuning XSBench example at small scales on a laptop for demonstration. 
For each application folder, there are two sub-folders: Summit and Theta each including the autotuning files for its platform.
All of them require the installation of ytopt autotuning framework (https://github.com/ytopt-team/ytopt) 
and MPI and OpenMP environments. XSBench-Laptop is the simplified version for autotuning XSBench on a laptop for a demo.

# Directory
```
AMG/	
    Autotuning AMG on Theta and Summit
SW4lite/
    Autotuning SW4lite on Theta and Summit
SWFFT/	
    Autotuning SWFFT on Theta and Summit  
XSBench/	
    Autotuning XSBench on Theta and Summit
XSBench-Laptop/	
    Autotuning XSBench on a laptop for demonstration
```

# ytopt Install instructions
The ytopt autotuning framework requires the following components: ConfigSpace, scikit-optimize, autotune, and ytopt. This installation should be very quick.

* We recommend creating isolated Python environments on your local machine using [conda](https://docs.conda.io/projects/conda/en/latest/index.html), for example, create a conda environment yt as follows:

```
conda create --name yt python=3.7
conda activate yt
```

* Create a directory for ytopt installation as follows:
```
mkdir yt
cd yt
```

* Install [ConfigSpace](https://github.com/ytopt-team/ConfigSpace.git):
```
git clone https://github.com/ytopt-team/ConfigSpace.git
cd ConfigSpace
pip install -e .
cd ..
```

* Install [scikit-optimize](https://github.com/ytopt-team/scikit-optimize.git):
```
git clone https://github.com/ytopt-team/scikit-optimize.git
cd scikit-optimize
pip install -e .
cd ..
```

* Install [autotune](https://github.com/ytopt-team/autotune.git):
```
git clone -b version1 https://github.com/ytopt-team/autotune.git
cd autotune
pip install -e . 
cd ..
```

* Install [ytopt](https://github.com/ytopt-team/ytopt.git):
```
git clone -b main https://github.com/ytopt-team/ytopt.git
cd ytopt
pip install -e .
cd ..
```
* If needed, install packaging 
```
pip install packaging
```

* If needed, downgrade the protobuf package to 3.20.x or lower
```
pip install protobuf==3.20
```

After this, the conda environment yt is installed successfully. 

# Instructions for testing the autotuning framework on a laptop 
Follow the ytopt installation instructions to install ytopt on a laptop such as a Macbook Pro. Aussume that MPI and OpenMP programming environments are installed and supported already (use "brew install open-mpi" to install MPI; use "brew install libomp" to install OpenMP). 

Download the file xsbench-mpi.zip under the folder XSBench-Laptop, then unzip the file to create the foler xsbench-mpi. Do the following steps:
```
cd xsbench-mpi
* If you want to change the compiler mpicc (default), edit the file plopper/plopper.py. 
cd xsbench
* make sure to start the ytopt conda environemnt yt before running a test
conda activate yt
* use the run script run.bat to autotune XSBench
./run.bat
```
After it is finished, one performance file results.csv is generated. The file looks like 
```
p0,p1,p2,p3,p4,p5,objective,elapsed_sec
2,10,#pragma omp parallel for,cores,close,static,45.825,56.204030990600586
6,128, ,cores,master,dynamic,24.861,88.20706510543823
3,100,#pragma omp parallel for,sockets,master,dynamic,33.733,128.20559000968933
6,256,#pragma omp parallel for,sockets,spread,static,24.38,160.20615696907043
8,400, ,threads,master,static,22.516,190.19838786125183
```
where p0,p1,p2,p3,p4,p5 are the tunable parameters; objective stands for the application execution time (in seconds); and elapsed_sec stands for the wall-clock time.
See the details about the autotuning scripts from the link https://github.com/ytopt-team/autotune/tree/master/Benchmarks/ECP-Apps/XSBench-Laptop/xsbench-mpi/xsbench.
