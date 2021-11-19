#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX environment variable not set"
  exit 1
fi


"${CLANG_PREFIX}/bin/clang" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native \
  -mllvm -polly -mllvm -polly-reschedule=0 -mllvm -polly-pattern-matching-based-opts=0 -mllvm -debug-only=polly-ast -fopenmp -mllvm -polly-scheduling=static -mllvm -polly-omp-backend=LLVM \
  *.c -o "${SCRIPTPATH}/${BASENAME}" \
  -I"/usr/include/x86_64-linux-gnu/mpich" -lmpich -lm -fcommon -DPOLYBENCH_TIME \
  -mllvm -polly-only-func=stencil7_calc_all
  

LD_LIBRARY_PATH="${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src:${LD_LIBRARY_PATH}" time "${SCRIPTPATH}/${BASENAME}" --nx 94 --ny 94 --nz 94 --max_blocks 128 --uniform_refine 1 --num_refine 2
