clang -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 Main.c Materials.c XSutils.c -o XSBench -lm -L/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib
