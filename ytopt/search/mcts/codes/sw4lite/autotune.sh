#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX environment variable not set"
  exit 1
fi

(cd "${SCRIPTPATH}" && python3 "${ROOTPATH}/mctree.py" --packing-arrays=a_mu,a_lambda,a_u,a_lu,a_strx,a_stry,a_strz,a_acof,a_bope,a_ghcof autotune --polybench-time \
  --exec-args "${SCRIPTPATH}/LOH.1-h100.in" --ld-library-path="${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  "${CLANG_PREFIX}/bin/clang++" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native \
  *.C *.c -o "${BASENAME}" \
  -I"/usr/include/x86_64-linux-gnu/mpich" -lmpich -llapack -lm -mllvm -polly-ignore-aliasing \
  -DSW4_CROUTINES -mllvm -polly-only-func=rhs4sg_rev_kernel \
  )
