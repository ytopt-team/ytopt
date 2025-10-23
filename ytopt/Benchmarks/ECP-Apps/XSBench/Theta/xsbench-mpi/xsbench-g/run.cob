#!/bin/bash

module use -a /projects/intel/geopm-home/modulefiles
module unload darshan
module load geopm/1.x
module load miniconda-3/latest

source activate yt
python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=5 --learner RF
conda deactivate
