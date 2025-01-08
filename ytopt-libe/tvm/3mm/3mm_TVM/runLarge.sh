#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --job-name=ytopt
#SBATCH --output=slurmLog/out/gpu-matmul-tvm.%j.out
#SBATCH --error=slurmLog/err/gpu-matmul-tvm.%j.err
#SBATCH --gres=gpu:1

source /home/pparamasivam/anaconda3/bin/activate
conda activate tvm

python tvm3MM_Baseline.py --size=L
python tvm3MM_GATuner.py --size=L
python tvm3MM_GridSearchTuner.py --size=L
python tvm3MM_RandTuner.py --size=L
python tvm3MM_XGBTuner.py --size=L
