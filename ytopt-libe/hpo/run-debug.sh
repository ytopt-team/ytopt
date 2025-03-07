#!/bin/bash

# set the total number of nodes
let nnds=1
# set the number of nodes for the MPI ranks per run
let nranks=1
# set the number of workers (number of nodes/nranks plus 1)
let nws=2
# set the total number of gpus per node
let nr=1
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=5000

#--- process processexe.pl to change the number of nodes (no change)
./processcp.pl ${nranks} 
./plopper.pl plopper.py ${appto}

#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash
#PBS -l select=${nnds}:ncpus=${nr}:ngpus=${nr}:system=polaris
#PBS -l place=scatter
#PBS -l walltime=72:00:00
#PBS -l filesystems=home:grand
#PBS -q preemptable
# debug 
#PBS -A EE-ECP

export PYTHONNOUSERSITE=1

export GPU_SUPPORT_ENABLED=1
export NCCL_COLLNET_ENABLE=1

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

# Activate conda environment
module use /soft/modulefiles
module load conda
conda activate ytune

# Launch libE
cd \$PBS_O_WORKDIR
python \$EXE \$COMMS \$NWORKERS --learner=RF --max-evals=128 > out.txt 2>&1
EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job
