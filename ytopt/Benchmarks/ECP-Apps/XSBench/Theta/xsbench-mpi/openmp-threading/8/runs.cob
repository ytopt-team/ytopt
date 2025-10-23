#!/bin/bash

let nnds=8
let nomp=64
let rpn=1
#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash
#COBALT -n ${nnds} -t 60 -O runs${nnds}x${nomp} -qdebug-cache-quad -A EE-ECP

export OMP_NUM_THREADS=${nomp}

aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench > out.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench > out2.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench > out3.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench > out4.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench > out5.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench -m event > mout.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench -m event > mout2.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench -m event > mout3.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench -m event > mout4.txt
aprun -n $((nnds*rpn)) -N ${rpn} --cc=depth -d ${nomp} -j 1 ./XSBench -m event > mout5.txt

EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job
