#!/bin/bash

let nnds=4096
let nomp=64
let rpn=1
#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash
#COBALT -n ${nnds} -t 30 -O runs${nnds}x${nomp} --attrs mcdram=cache:numa=quad -A Intel

export OMP_NUM_THREADS=${nomp}

module load intel
module use -a /projects/intel/geopm-home/modulefiles
module unload darshan
module load geopm/1.x

           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm.report -- ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm2.report -- ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out2.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm3.report -- ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out3.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm4.report -- ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out4.txt
           geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm5.report -- ../amg2013 -laplace -P 16 16 16 -n 100 100 100 > out5.txt

EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
qsub batch.job
