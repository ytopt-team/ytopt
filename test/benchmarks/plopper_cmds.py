MAT_GCC_CMD = (
    "{} -std=gnu99 -Wall -flto  -fopenmp -DOPENMP {} -O3"
    + " -o {} {} {} /Materials.c {}/XSutils.c -I{} -lm -L${CONDA_PREFIX}/lib"
)

CONV_CLANG_CMD = (
    "clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75"
    + " -DEXTRALARGE_DATASET -DPOLYBENCH_TIME -I{} -I{} {} {}/polybench.c -o {}"
    + " -lm -g -I/soft/compilers/cuda/cuda-11.4.0/include -L/soft/compilers/cuda/cuda-11.4.0/lib64"
    + " -Wl,-rpath=/soft/compilers/cuda/cuda-11.4.0/lib64 -lcudart_static -ldl -lrt -pthread"
)

GPP_BLOCK_CMD = "g++ {}/mmm_block.cpp -DBLOCK_SIZE={} -o {}"

CUDA_FLAGS = (
    "-I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64"
    + "-Wl -rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread"
)

CLANG_CUDA_CMD = "invoke_test {} clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march={} {} {}"

COMMON_FLAGS = (
    "-DLARGE_DATASET -DPOLYBENCH_TIME -I{} -I{} {} {}/polybench.c -o {} -lm -g "
)

XL_CUDA_FLAGS = (
    "-I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64 -Wl"
    + " -rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread"
)

XL_CUDA_CMD = "invoke_test {} /usr/tce/packages/xl/xl-2021.03.11/bin/xlc++ -qsmp -qoffload -qtgtarch={} {} {}"


CLANG_CUDA_NVPT_CMD = (
    "invoke_test clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_60 {} "
    + "-I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64 -Wl"
    + " -rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread"
)

CMD2 = "invoke_test {}/time_benchmark.sh {}"