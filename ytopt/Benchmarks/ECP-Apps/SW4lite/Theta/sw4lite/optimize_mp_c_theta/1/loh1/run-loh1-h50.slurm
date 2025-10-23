#!/bin/bash
#SBATCH -N 4
#SBATCH -p debug
#SBATCH -A m2545
#SBATCH -S 4     # Special cores per nodes (4 are idle)
#SBATCH -t 00:30:00 
#SBATCH -C knl,quad,cache
##SBATCH -C knl,quad,flat

numnodes=4

for num_mpi_per_node in 64 32 16 8; do
#for num_mpi_per_node in 4 2 1; do

# developer branch
#  num_mpi_per_node=64

total_num_mpi=$(( ${numnodes}*${num_mpi_per_node} ))
echo "num_mpi_per_node=" $num_mpi_per_node

# Number of ranks x threads is constant
# hc = hardware cores
numhc=$(echo ${num_mpi_per_node} | awk '{print 64/$1}')
echo "numhc=" $numhc
# hyper-threads
numht=1
# logical cores = OMP_NUM_TASK
num_threads_per_mpi=$(( ${numhc}*${numht} ))
# srun reservation logical cores per mpi-task  (-c arg)
numlc=$(( ${numhc}*4 ))

echo "Running on ${numnodes} nodes with ${total_num_mpi} MPI ranks and OMP_NUM_THREADS=${num_threads_per_mpi}."

# Intel OpenMP runtime parameters
export OMP_NUM_THREADS=${num_threads_per_mpi}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Run the job with this MPI + OpenMP configuration
# quad modes
MPI_COMMAND="srun -N ${numnodes} -n ${total_num_mpi} -c ${numlc} --cpu_bind=cores" 
# flat mode needs the numactl command
#MPI_COMMAND="srun -N ${numnodes} -n ${total_num_mpi} -c ${numlc} --cpu_bind=cores numactl -m 1" 

# 1st case
CASE=LOH.1-h50
RUN_COMMAND="./sw4lite_mpc ${CASE}.in"
${MPI_COMMAND} ${RUN_COMMAND}  >& ${CASE}-sw4-${total_num_mpi}mpi-${num_threads_per_mpi}threads.out

done
