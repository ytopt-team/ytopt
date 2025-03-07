#!/bin/bash

# set the total number of nodes
let nnds=1
# set the number of nodes for the MPI ranks per run
let nranks=1
# set the total number of gpus per node
let nr=1

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

# Activate conda environment
module use /soft/modulefiles
module load conda
conda activate ytune

# Launch libE
cd \$PBS_O_WORKDIR
python -m ytopt.search.ambs --evaluator ray --problem problem.Problem  --learner=RF --max-evals=128 > out.txt 2>&1
EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job

