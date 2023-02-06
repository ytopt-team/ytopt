MAT_GCC_CMD = (
    "{compiler} -std=gnu99 -Wall -flto  -fopenmp -DOPENMP {mpi_macro} -O3"
    + " -o {tmpbinary} {interimfile} {kernel_dir} /Materials.c {kernel_dir}/XSutils.c -I{kernel_dir} -lm -L${CONDA_PREFIX}/lib"
)

CONV_CLANG_CMD = (
    "clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_75"
    + " -DEXTRALARGE_DATASET -DPOLYBENCH_TIME -I{utilities_dir} -I{kernel_dir} {interimfile} {utilities_dir}/polybench.c -o {tmpbinary}"
    + " -lm -g -I/soft/compilers/cuda/cuda-11.4.0/include -L/soft/compilers/cuda/cuda-11.4.0/lib64"
    + " -Wl,-rpath=/soft/compilers/cuda/cuda-11.4.0/lib64 -lcudart_static -ldl -lrt -pthread"
)

GPP_BLOCK_CMD = (
    "g++ {kernel_dir}/mmm_block.cpp -DBLOCK_SIZE={block_size} -o {tmpbinary}"
)

CUDA_FLAGS = (
    "-I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64"
    + " -Wl -rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread"
)

CLANG_CUDA_CMD = (
    "invoke_test {system} clang -O2 -fopenmp -fopenmp-targets=nvptx64"
    + " -Xopenmp-target -march={gpuarch} {commonflags} {cuda_flags}"
)

COMMON_FLAGS = (
    "-DLARGE_DATASET -DPOLYBENCH_TIME -I{utilities_dir} -I{kernel_dir} {interimfile} {utilities_dir}/polybench.c"
    + " -o {tmpbinary} -lm -g "
)

XL_CUDA_FLAGS = (
    "-I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64 -Wl"
    + " -rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread"
)

XL_CUDA_CMD = (
    "invoke_test {system} /usr/tce/packages/xl/xl-2021.03.11/bin/xlc++"
    + " -qsmp -qoffload -qtgtarch={gpuarch} {commonflags} {cuda_flags}"
)

CLANG_CUDA_NVPT_CMD = (
    "invoke_test clang -O2 -fopenmp -fopenmp-targets=nvptx64 -Xopenmp-target -march=sm_60 {commonflags} "
    + "-I/usr/tce/packages/cuda/cuda-10.1.168/include -L/usr/tce/packages/cuda/cuda-10.1.168/lib64 -Wl"
    + " -rpath=usr/tce/packages/cuda/cuda-10.1.168/lib64 -lcudart_static -ldl -lrt -pthread"
)

CMD2 = "invoke_test {utilities_dir}/time_benchmark.sh {tmpbinary}"
