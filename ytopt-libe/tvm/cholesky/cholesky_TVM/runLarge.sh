#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=6:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-tvm.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-tvm.%j.err
#SBATCH --gres=gpu:1

source /home/pparamasivam/anaconda3/bin/activate
conda activate tvm

python Cholesky_Baseline.py --size=L
python Cholesky_GATuner.py --size=L
python Cholesky_GridSearch.py --size=L
python Cholesky_RandTuner.py --size=L
python Cholesky_XGBTuner.py --size=L
