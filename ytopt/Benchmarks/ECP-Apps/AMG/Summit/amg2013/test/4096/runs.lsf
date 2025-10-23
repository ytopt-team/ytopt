#!/bin/bash

let nnds=4096
#--- process the submit script

#-----This part creates a submission script---------
cat >batch.job <<EOF
# Begin LSF directives
#BSUB -P MED106
#BSUB -J ytopt
#BSUB -o output.log
#BSUB -e output.err
#BSUB -alloc_flags gpumps
#BSUB -W 30
#BSUB -nnodes ${nnds}

export OMP_NUM_THREADS=168
jsrun -n ${nnds} -a 1 -g 0 -c 42 -bpacked:42 -dpacked ../amg2013 -laplace -P 16 16 16 -n 100 100 100  > out.txt
jsrun -n ${nnds} -a 1 -g 0 -c 42 -bpacked:42 -dpacked ../amg2013 -laplace -P 16 16 16 -n 100 100 100  > out2.txt
jsrun -n ${nnds} -a 1 -g 0 -c 42 -bpacked:42 -dpacked ../amg2013 -laplace -P 16 16 16 -n 100 100 100  > out3.txt
jsrun -n ${nnds} -a 1 -g 0 -c 42 -bpacked:42 -dpacked ../amg2013 -laplace -P 16 16 16 -n 100 100 100  > out4.txt
jsrun -n ${nnds} -a 1 -g 0 -c 42 -bpacked:42 -dpacked ../amg2013 -laplace -P 16 16 16 -n 100 100 100  > out5.txt

EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
bsub  batch.job
