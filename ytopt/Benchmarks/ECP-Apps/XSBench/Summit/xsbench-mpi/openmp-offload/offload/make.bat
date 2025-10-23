module load nvhpc
mpicc -mp=gpu -std=gnu99 -Wall -DMPI -fopenmp -O3 Main.c XSutils.c Materials.c -o XSBench -lm
