#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX environment variable not set"
  exit 1
fi
  
cd "${SCRIPTPATH}"
"${CLANG_PREFIX}/bin/clang" -DLARGE_DATASET -DPOLYBENCH_TIME polybench.c -I"/home/jkoo/anaconda3/envs/ytune_mc/include" -L"/home/jkoo/anaconda3/envs/ytune_mc/lib" -lblas gemm_mkl.c -o "${BASENAME}_mkl"

python "${ROOTPATH}/mctree_o3p_mkl.py" "${SCRIPTPATH}" "${BASENAME}_mkl"
