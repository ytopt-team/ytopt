#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-tvm.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-tvm.%j.err
#SBATCH --gres=gpu:1
#SBATCH --account=EE-ECP

source /home/pparamasivam/anaconda3/bin/activate
conda activate tvm

python LU_Baseline.py --size=L
python LU_GATuner.py --size=L
python LU_GridSearch.py --size=L
python LU_RandTuner.py --size=L
python LU_XGBTuner.py --size=L
