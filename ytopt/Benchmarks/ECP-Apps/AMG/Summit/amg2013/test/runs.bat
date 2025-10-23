#time mpirun -np 4 amg2013 -P 2 2 1 -n 80 80 80 -printstats
#time mpirun -np 8 amg2013 -pooldist 1 P 1 1 1 -r 48 48 48 -printstats
#time mpirun -np 8 amg2013 -laplace -P 2 2 2 -n 80 80 80 -printstats
#time mpirun -np 4 amg2013 -laplace -n 300 300 300 -P 2 2 1 -printstats
export OMP_NUM_THREADS=8
time mpirun -np 1 amg2013 -laplace -n 300 300 300 -P 1 1 1 -printstats > p1t8.txt
export OMP_NUM_THREADS=4
time mpirun -np 1 amg2013 -laplace -n 300 300 300 -P 1 1 1 -printstats > p1t4.txt
export OMP_NUM_THREADS=2
time mpirun -np 1 amg2013 -laplace -n 300 300 300 -P 1 1 1 -printstats > p1t2.txt
export OMP_NUM_THREADS=1
time mpirun -np 1 amg2013 -laplace -n 300 300 300 -P 1 1 1 -printstats > p1t1.txt

