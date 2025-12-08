#!/bin/bash
#This script is for running an app on a laptop using mpirun without any scheduler

# set the number of nodes
let nnds=32
# set the number of nodes for the MPI ranks per run
let nranks=1
# set the number of workers (nnds/nranks plus 1)
let nws=33
# set the total number of gpus per node
let nr=1
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=5000
# set number of evaluations
let nevals=512

#--- process processexe.pl to change the number of nodes (no change)

# set the MPI ranks per run
./processcp.pl ${nranks}

# set application timeout
./plopper.pl plopper.py ${appto}

#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash -x

#PBS -l select=${nnds}:system=polaris
#PBS -N ytopt-julia
#PBS -l place=scatter
#PBS -l walltime=06:00:00
#PBS -l filesystems=home:grand
#PBS -A PIONEER
#PBS -q prod

module use /soft/modulefiles
module load conda
module use /eagle/EE-ECP/julia_depot/modulefiles/
module load julia/1.11

# change the code path to your own one
CODE_PATH="/home/wuxf/worker/PhasorNetworks.jl"

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
export CONDA_ENV_NAME=ytune

export PMI_NO_FORK=1 # Required for python kills on Theta

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate \$CONDA_ENV_NAME

# Launch libE
cd $CODE_PATH/hpo

python \$EXE \$COMMS \$NWORKERS --learner=RF --max-evals=${nevals} > out.txt 2>&1
EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job
