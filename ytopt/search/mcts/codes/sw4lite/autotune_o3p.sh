#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX envirnment variable not set"
  exit 1
fi
module add openmpi
export LD_LIBRARY_PATH="${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src:/soft/libraries/mpi/openmpi/2.1.6/lib:/home/jkoo/anaconda3/envs/ytune_mc/lib:${LD_LIBRARY_PATH}"
  
cd "${SCRIPTPATH}"
"${CLANG_PREFIX}/bin/clang++" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
 -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native -mllvm -polly -mllvm -polly-parallel -fopenmp -mllvm -polly-omp-backend=LLVM -mllvm -polly-scheduling=static -L"${CLANG_PREFIX}/lib" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -DLARGE_DATASET -DPOLYBENCH_TIME *.C *.c  -o "${BASENAME}_o3p" -I"/soft/libraries/mpi/openmpi/2.1.6/include" -L"/soft/libraries/mpi/openmpi/2.1.6/lib" -lmpi -I"/home/jkoo/anaconda3/envs/ytune_mc/include" -L"/home/jkoo/anaconda3/envs/ytune_mc/lib" -llapack -lm -mllvm -polly-ignore-aliasing -DSW4_CROUTINES

perf stat -- "./${BASENAME}_o3p"
python "${ROOTPATH}/mctree_o3p_ecp.py" "${SCRIPTPATH}" "${BASENAME}_o3p"