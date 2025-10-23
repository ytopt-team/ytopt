module load nvhpc  
mpicc -mp=gpu -std=gnu99 -Wall -fopenmp -DMPI -O3 Main.c XSutils.c Materials.c -o XSBench -lm
