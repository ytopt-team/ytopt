#clang -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -mllvm -polly-reschedule=0 -mllvm -polly-postopts=0 -ffast-math -march=native mm.c Materials.c XSutils.c -o XSB -lm 
clang -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math -march=native mm.c Materials.c XSutils.c -o XSB -lm 
#clang -std=gnu99 -Wall -flto  -fopenmp -DOPENMP -O3 mm.c Materials.c XSutils.c -o XSB -lm 
