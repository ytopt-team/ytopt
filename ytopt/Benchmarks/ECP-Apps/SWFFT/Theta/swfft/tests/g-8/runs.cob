#!/bin/bash

let nnds=8
let nomp=64
let rpn=1
#-----This part creates a submission script---------
cat >batch2.job <<EOF
#!/bin/bash
#COBALT -A EE-ECP -n ${nnds} -t 60 -O runs${nnds}x${rpn}x${nomp} -qdebug-cache-quad  

export OMP_NUM_THREADS=${nomp}

module load intel
module use -a /projects/intel/geopm-home/modulefiles
module unload darshan
module load geopm/1.x

geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm.report -- ../TestDfft 2 512 > out.txt
geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm2.report -- ../TestDfft 2 512 > out2.txt
geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm3.report -- ../TestDfft 2 512 > out3.txt
geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm4.report -- ../TestDfft 2 512 > out4.txt 
geopmlaunch aprun -n $((nnds*rpn)) -N ${rpn} --geopm-ctl=pthread --geopm-report gm5.report -- ../TestDfft 2 512 > out5.txt 


EOF
#-----This part submits the script you just created--------------
chmod +x batch2.job
qsub batch2.job
