#clang -DEXTRALARGE_DATASET -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math -march=native polybench.c lu.c -o lu 
clang -DLARGE_DATASET -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math -march=native polybench.c lu.c -o lu 
#clang -DHUGE_DATASET -DPOLYBENCH_TIME -std=c99 -fno-unroll-loops -O3 -mllvm -polly -mllvm -polly-process-unprofitable -mllvm -polly-use-llvm-names -ffast-math -march=native polybench.c lu.c -o lu 
