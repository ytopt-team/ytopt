This directory includes all files for autotuning ECP proxy app XSBench (https://github.com/ANL-CESAR/XSBench) using ytopt-libensemble on a laptop, ALCF Theta and OLCF Crusher, and using ytopt only on ALCF Theta.

# Directory
```
laptop/	
    Autotuning XSBench on a laptop for demonstration
theta/
    Autotuning XSBench on ALCF Theta
ytopt/	
    Autotuning XSBench using ytopt only on ALCF Theta
crusher/	
    Autotuning XSBench on OLCF Crusher 

```

# Instructions for testing the autotuning framework on a laptop 
Follow the ytopt-libensemble installation instructions to install ytopt and libensemble on a laptop such as a Macbook Pro. Aussume that MPI and OpenMP programming environments are installed and supported already (use "brew install open-mpi" to install MPI; use "brew install libomp" to install OpenMP). 

Do the following steps:
```
git clone https://github.com/ytopt-team/ytopt-libensemble.git
cd ytopt-libensemble
cd ytopt-libe-xsbench
cd laptop

* If you want to change the compiler mpicc (default), edit the file plopper.py. 
* Make sure to create the conda environemnt ytune before running a test
* Modify the run script runs.sh with the proper conda environment, number of wokers, MPI ranks, and the application timeout
* Then, use the run script to autotune XSBench 

./runs.sh
```
After it is finished, one performance file results.csv is generated. The file looks like 
```
p0,p1,p2,p3,p4,RUNTIME,elapsed_sec
4,100, ,cores,close,3.242,5.355260848999023
8,64, ,sockets,master,1.913,8.534861087799072
4,20, ,threads,master,3.323,8.549333095550537
8,40, ,threads,master,1.54,11.50633192062378
8,10,#pragma omp parallel for,threads,spread,1.614,14.497555017471313
2,40,#pragma omp parallel for,sockets,spread,5.737,15.974946022033691
8,160,#pragma omp parallel for,threads,master,1.333,17.22815704345703
8,128,#pragma omp parallel for,threads,master,1.313,18.940433979034424
```
where p0,p1,p2,p3,p4 are the tunable parameters; objective stands for the application execution time (in seconds); and elapsed_sec stands for the wall-clock time. For the diagnosis purpose, look at the log files (*.log) or text files (*.txt) for any error under the current directory.
