mpicc -std=gnu99 -Wall -flto -dynamic -fopenmp -DOPENMP -DMPI -O3 Main.c Materials.c XSutils.c -o XSBench -lm -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
