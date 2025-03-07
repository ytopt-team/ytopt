#!/bin/bash
#This script is for running an app on a laptop using mpirun without any scheduler

# find the conda path
cdpath=$(conda info | grep -i 'base environment')
arr=(`echo ${cdpath}`)
cpath="$(echo ${arr[3]})/etc/profile.d/conda.sh"

#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash -x

# Name of Conda environment
export CONDA_ENV_NAME=ytune


# Activate conda environment
source $cpath
export PYTHONNOUSERSITE=1
conda activate \$CONDA_ENV_NAME

# Launch ytopt

python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=2 --learner RF

# print out the best configureation
python findMin.py

EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
./batch.job

