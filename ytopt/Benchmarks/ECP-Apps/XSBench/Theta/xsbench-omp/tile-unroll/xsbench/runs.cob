#!/bin/bash

let nnds=1
#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash
#COBALT -n ${nnds} -t 60 -O runs${nnds} -qdebug-cache-quad -A EE-ECP

module load miniconda-3/latest
source activate yt

python3 -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=200 --learner RF

conda deactivate

EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job
