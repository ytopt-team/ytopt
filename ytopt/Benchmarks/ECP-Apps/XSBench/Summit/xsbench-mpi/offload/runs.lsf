#!/bin/bash

let nnds=4096
#--- process exe.pl to change the number of nodes
./processcp.pl ${nnds}

#-----This part creates a submission script---------
cat >batch.job <<EOF
# Begin LSF Directives
#BSUB -P AST136
#BSUB -W 00:30
#BSUB -nnodes ${nnds}
#BSUB -alloc_flags gpumps
#BSUB -J ytopt
#BSUB -o ytopt.%J.out
#BSUB -e ytopt.%J.err

module load ibm-wml-ce
conda activate yt

python3 -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=200 --learner RF

EOF
#-----This part submits the script you just created--------------
bsub  batch.job
