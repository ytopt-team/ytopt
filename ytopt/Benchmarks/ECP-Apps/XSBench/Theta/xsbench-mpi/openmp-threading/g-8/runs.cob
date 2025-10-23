#!/bin/bash

let nnds=8
let nomp=64
let rpn=1
#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash
#COBALT -n ${nnds} -t 60 -O runs${nnds}x${nomp} -qdebug-cache-quad -A EE-ECP

export OMP_NUM_THREADS=${nomp}

module load intel
module use -a /projects/intel/geopm-home/modulefiles
module unload darshan
module load geopm/1.x

           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm.report -- ../XSBench -m event > mout.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm2.report -- ../XSBench -m event > mout2.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm3.report -- ../XSBench -m event > mout3.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm4.report -- ../XSBench -m event > mout4.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm5.report -- ../XSBench -m event > mout5.txt

EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job
