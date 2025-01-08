#!/bin/bash

module swap PrgEnv-intel PrgEnv-gnu
module load gcc 
module load miniconda-3/latest

source activate yt
python3 -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=5 --learner RF
