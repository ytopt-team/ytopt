#!/bin/bash

# set the number of nodes
let nnds=4
# set the number of nodes for the MPI ranks per run
let nranks=4
# set the number of workers (number of nodes/nranks plus 1)
let nws=2
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=500

#--- process processexe.pl to change the number of nodes (no change)
# set the MPI ranks per run
./processcp.pl ${nranks}

# set application timeout
./plopper.pl plopper.py ${appto}

# find the conda path
cdpath=$(conda info | grep -i 'base environment')
arr=(`echo ${cdpath}`)
cpath="$(echo ${arr[3]})/etc/profile.d/conda.sh"

#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash -x
##SBATCH -A MED106_crusher
##SBATCH -A AST136_crusher
##SBATCH -A CSC383
#SBATCH -A CSC383_crusher
#SBATCH -J ytopt
#SBATCH -o %x-%j.out
#SBATCH -t 00:50:00
#SBATCH -p batch
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=64
##SBATCH --gpus-per-task=${nr}
##SBATCH --gpu-bind=closest
#SBATCH --threads-per-core=2
#SBATCH -N ${nnds}

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the nodes in the job available (using exe.pl)

# Name of calling script
export EXE=run_ytopt.py

# Communication Method
export COMMS="--comms local"

# Number of workers. For multiple nodes per worker, have nworkers be a divisor of nnodes, then add 1
# e.g. for 2 nodes per worker, set nnodes = 12, nworkers = 7
export NWORKERS="--nworkers ${nws}"  # extra worker running generator (no resources needed)
# Adjust exe.pl so workers correctly use their resources

# Name of Conda environment
#export CONDA_ENV_NAME=ytune
export CONDA_ENV_NAME=ytl

export PMI_NO_FORK=1 # Required for python kills on Theta

# Unload Theta modules that may interfere with job monitoring/kills
module unload trackdeps
module unload darshan
module unload xalt

#load needed modules
#source /ccs/home/wuxf/anaconda3/etc/profile.d/conda.sh
source $cpath
module load PrgEnv-amd/8.3.3
module load cray-hdf5/1.12.0.7
module load cmake
module load craype-accel-amd-gfx90a
module load rocm/4.5.2
module load cray-mpich/8.1.14
#export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_IGNORE_SRAMECC_MISREPORT=1
## These must be set before compiling so the executable picks up GTL
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate \$CONDA_ENV_NAME

# Launch libE
python \$EXE \$COMMS \$NWORKERS --learner=RF --max-evals=16 > out.txt 2>&1
EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
sbatch batch.job
