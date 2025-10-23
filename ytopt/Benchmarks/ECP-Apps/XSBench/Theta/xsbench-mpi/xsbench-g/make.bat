module load xl
cc -std=gnu99 -Wall -flto -dynamic -fopenmp -DOPENMP -DMPI -O3 Main.c Materials.c XSutils.c -o XSBench -lm 
