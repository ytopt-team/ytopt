#!/bin/bash

let nnds=1024
let nomp=64
let rpn=1
#-----This part creates a submission script---------
cat >batch2.job <<EOF
#!/bin/bash
#COBALT -A EE-ECP -n ${nnds} -t 60 -O runs${nnds}x${rpn}x${nomp} --attrs mcdram=cache:numa=quad

export OMP_NUM_THREADS=${nomp}

           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 8 8 -n 100 100 100 > out.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 8 8 -n 100 100 100 > out2.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 8 8 -n 100 100 100 > out3.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 8 8 -n 100 100 100 > out4.txt
           aprun -n $((nnds*rpn)) -N ${rpn} -cc depth -d 64 -j 1 ../amg2013 -laplace -P 16 8 8 -n 100 100 100 > out5.txt


EOF
#-----This part submits the script you just created--------------
chmod +x batch2.job
qsub batch2.job
