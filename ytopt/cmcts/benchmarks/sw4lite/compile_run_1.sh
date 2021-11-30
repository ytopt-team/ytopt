#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX environment variable not set"
  exit 1
fi
module add openmpi
CFLAGS=(-I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native \
  -I"/soft/libraries/mpi/openmpi/2.1.6/include" -L"/soft/libraries/mpi/openmpi/2.1.6/lib" -lmpi \
  -I"/home/jkoo/anaconda3/envs/ytune_mc/include" -L"/home/jkoo/anaconda3/envs/ytune_mc/lib" -llapack -lm \
  -I"${SCRIPTPATH}" -DSW4_CROUTINES -DDEBUG -DPOLYBENCH_TIME -mllvm -polly -mllvm -polly-position=early -mllvm -polly-only-func=rhs4sg_rev_kernel  -mllvm -polly-process-unprofitable -mllvm -polly-allow-nonaffine-branches -mllvm -polly-print-instructions -mllvm -polly-use-llvm-names -mllvm -polly-process-unprofitable -mllvm -debug-only=polly-detect,polly-scops -mllvm -polly-ignore-aliasing)

"${CLANG_PREFIX}/bin/clang++" ${CFLAGS[*]} -c *.c *.C
rm -f rhs4sg_rev.o

echo ${CFLAGS[*]} `pwd`/rhs4sg_rev.C -o "${BASENAME}"
"${CLANG_PREFIX}/bin/clang++" ${CFLAGS[*]}  rhs4sg_rev.C *.o  -o "${BASENAME}"

"${SCRIPTPATH}/${BASENAME}" LOH.1-h100.in
