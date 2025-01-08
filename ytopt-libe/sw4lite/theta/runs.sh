#!/bin/bash

# set the number of nodes
let nnds=8
# set the total number of MPI ranks
let nranks=2
# set the number of workers (number of nodes/nranks plus 1)
let nws=5
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=300

#--- process processexe.pl to change the number of nodes (no change)
./processcp.pl ${nranks}
./plopper.pl plopper.py ${appto}

#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash -x
#COBALT -t 00:60:00
#COBALT -n ${nnds}
#COBALT -A EE-ECP
#COBALT -q debug-flat-quad

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
export CONDA_ENV_NAME=yt

export PMI_NO_FORK=1 # Required for python kills on Theta

# Unload Theta modules that may interfere with job monitoring/kills
module load miniconda-3/latest
module unload trackdeps
module unload darshan
module unload xalt

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate \$CONDA_ENV_NAME

# Launch libE
python \$EXE \$COMMS \$NWORKERS --learner=RF --max-evals=64 > out.txt 2>&1
EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job
