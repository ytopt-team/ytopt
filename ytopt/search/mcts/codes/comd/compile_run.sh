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
  *.c -o "${BASENAME}" \
  -I"${SCRIPTPATH}" -DDOUBLE -DNDEBUG -DPOLYBENCH_TIME -lm -mllvm -polly -mllvm -polly-position=early -mllvm -debug-only=polly-detect,polly-scops,polly-isl -mllvm -polly-only-func=ljForce_kernel -mllvm -polly-allow-nonaffine -g -mllvm -polly-process-unprofitable -mllvm -polly-allow-nonaffine-branches -mllvm -polly-print-instructions -mllvm -polly-use-llvm-names

"${SCRIPTPATH}/${BASENAME}"
