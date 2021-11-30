#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX envirnment variable not set"
  exit 1
fi

export LD_LIBRARY_PATH="${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src:${LD_LIBRARY_PATH}"
  
cd "${SCRIPTPATH}"
"${CLANG_PREFIX}/bin/clang" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native -mllvm -polly -mllvm -polly-parallel -fopenmp -mllvm -polly-omp-backend=LLVM -mllvm -polly-scheduling=static -L"${CLANG_PREFIX}/lib" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -DLARGE_DATASET -DPOLYBENCH_TIME polybench.c "${BASENAME}.c" -o "${BASENAME}_o3p" 

perf stat -- "./${BASENAME}_o3p"
python "${ROOTPATH}/mctree_o3p.py" "${SCRIPTPATH}" "${BASENAME}_o3p"