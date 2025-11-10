#!/bin/bash -x 

# set the number of nodes
let nnds=1
# set the total number of gpus per task
#let nr=1
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=500

#--- process processexe.pl to change the number of nodes (no change)
./processcp.pl ${nnds}
./plopper.pl plopper.py ${appto}

#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash -x
##SBATCH -A MED106_crusher
##SBATCH -A AST136_crusher
#SBATCH -A CSC383
#SBATCH -J ytopt
#SBATCH -o %x-%j.out
#SBATCH -t 08:00:00
#SBATCH -p batch
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=64
##SBATCH --gpus-per-task=${nr}
##SBATCH --ntasks-per-gpu=${nr}
##SBATCH --gpu-bind=closest
#SBATCH --threads-per-core=2
#SBATCH -N ${nnds}

source /ccs/home/wuxf/anaconda3/etc/profile.d/conda.sh
module load PrgEnv-amd/8.3.3
module load cray-hdf5/1.12.0.7
module load cmake
module load craype-accel-amd-gfx90a
module load rocm/4.5.2
module load cray-mpich/8.1.14
#export MPICH_GPU_SUPPORT_ENABLED=1
export HSA_IGNORE_SRAMECC_MISREPORT=1
## These must be set before compiling so the executable picks up GTL
export PE_MPICH_GTL_DIR_amd_gfx90a="-L${CRAY_MPICH_ROOTDIR}/gtl/lib"
export PE_MPICH_GTL_LIBS_amd_gfx90a="-lmpi_gtl_hsa"

conda activate ytune
python -m ytopt.search.ambs --evaluator ray --problem problem.Problem --max-evals=128 --learner RF
conda deactivate

EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
sbatch batch.job
