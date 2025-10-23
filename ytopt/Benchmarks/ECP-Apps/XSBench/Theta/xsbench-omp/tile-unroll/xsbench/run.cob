#!/bin/bash

module load miniconda-3/latest
source activate yt

python3 -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=2 --learner RF

conda deactivate
