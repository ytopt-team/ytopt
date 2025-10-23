# fftw
module load cray-libsci
module load cray-fftw

CC -O3 -fopenmp -Wall -DDFFT_TIMING=0 -I/opt/cray/pe/fftw/3.3.8.6/mic_knl/include -c -o TestDfft.o TestDfft.cpp
cc -O3 -fopenmp -Wall -Wno-deprecated -std=gnu99 -DDFFT_TIMING=0 -I/opt/cray/pe/fftw/3.3.8.6/mic_knl/include -c -o distribution.o distribution.c
CC -O3 -fopenmp -Wall -o TestDfft TestDfft.o distribution.o -fopenmp -L/opt/cray/pe/fftw/3.3.8.6/mic_knl/lib -lfftw3_omp -lfftw3 -lm
