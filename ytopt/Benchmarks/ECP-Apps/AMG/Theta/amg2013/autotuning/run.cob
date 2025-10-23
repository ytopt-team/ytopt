#!/bin/bash

module load miniconda-3/latest
source activate ytune

python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=100 --learner RF

conda deactivate
