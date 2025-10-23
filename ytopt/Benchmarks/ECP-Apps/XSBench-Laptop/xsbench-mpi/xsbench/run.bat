#!/bin/bash

let nnds=1
#--- process processexe.pl to change the number of MPI ranks
./processcp.pl ${nnds}

python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=5 --learner RF

