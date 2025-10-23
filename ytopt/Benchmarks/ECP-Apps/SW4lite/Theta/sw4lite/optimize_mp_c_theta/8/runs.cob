#!/bin/bash

let nnds=8
let nomp=64
let rpn=1
#-----This part creates a submission script---------
cat >batch2.job <<EOF
#!/bin/bash
#COBALT -A EE-ECP -n ${nnds} -t 60 -O runs${nnds}x${rpn}x${nomp} -qdebug-cache-quad  

export OMP_NUM_THREADS=${nomp}

           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d ${nomp} -j 1 ../sw4lite loh1/LOH.1-h50.in >out.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d ${nomp} -j 1 ../sw4lite loh1/LOH.1-h50.in >out2.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d ${nomp} -j 1 ../sw4lite loh1/LOH.1-h50.in >out3.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d ${nomp} -j 1 ../sw4lite loh1/LOH.1-h50.in >out4.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d ${nomp} -j 1 ../sw4lite loh1/LOH.1-h50.in >out5.txt


EOF
#-----This part submits the script you just created--------------
chmod +x batch2.job
qsub batch2.job
