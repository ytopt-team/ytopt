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
   -flegacy-pass-manager -O3 -march=native \
   polybench.c gemm_aocl.c -o "${SCRIPTPATH}/gemm_aocl" \
   -lblis -I"${SCRIPTPATH}" \
   -DLARGE_DATASET -UBLIS_ENABLE_OPENMP \
)

LD_LIBRARY_PATH="${SCRIPTPATH}/amd-blis/lib:${LD_LIBRARY_PATH}" time "${SCRIPTPATH}/gemm_aocl"
