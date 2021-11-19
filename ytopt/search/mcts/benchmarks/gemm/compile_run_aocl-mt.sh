#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX environment variable not set"
  exit 1
fi

("${CLANG_PREFIX}/bin/clang" -I"${SCRIPTPATH}/amd-blis/include" -L"${SCRIPTPATH}/amd-blis/lib" \
   -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
   -flegacy-pass-manager -O3 -march=native \
   polybench.c gemm_aocl.c -o "${SCRIPTPATH}/gemm_aocl-mt" \
   -fopenmp -lblis-mt -I"${SCRIPTPATH}" \
   -DLARGE_DATASET -DBLIS_ENABLE_OPENMP \
)

LD_LIBRARY_PATH="${SCRIPTPATH}/amd-blis/lib:${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src:${LD_LIBRARY_PATH}" time "${SCRIPTPATH}/gemm_aocl-mt"
