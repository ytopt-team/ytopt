This directory includes autotuning XSBench on a laptop (MacBook Pro). This example demonstrates how to autotune at a small scale. After the ytopt is installed on the laptop, download the file xsbench-mpi.zip which includes the folders xsbench (for autotuning scripts), plopper (for compiling and execution), and openmp-threading (original code from https://github.com/ANL-CESAR/XSBench). 

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


