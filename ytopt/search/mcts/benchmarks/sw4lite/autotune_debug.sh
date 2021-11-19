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
(cd "${SCRIPTPATH}" && python3 "${ROOTPATH}/mctree_bf.py" --packing-arrays=a_mu,a_lambda,a_u,a_lu,a_strx,a_stry,a_strz,a_acof,a_bope,a_ghcof autotune --keep --polybench-time --exec-args "${SCRIPTPATH}/LOH.1-h100.in" --ld-library-path="${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src:/soft/libraries/mpi/openmpi/2.1.6/lib:/home/jkoo/anaconda3/envs/ytune_mc/lib" \
 "${CLANG_PREFIX}/bin/clang++" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native \
  /gpfs/jlse-fs0/users/jkoo/run/mctree/codes/sw4lite/*.o /gpfs/jlse-fs0/users/jkoo/run/mctree/codes/sw4lite/rhs4sg_rev.C *.c -o "${BASENAME}" \
  -I"/soft/libraries/mpi/openmpi/2.1.6/include" -L"/soft/libraries/mpi/openmpi/2.1.6/lib" -lmpi -I"/home/jkoo/anaconda3/envs/ytune_mc/include" -L"/home/jkoo/anaconda3/envs/ytune_mc/lib" -llapack -lm -mllvm -polly-ignore-aliasing \
  -DSW4_CROUTINES -mllvm -polly-only-func=rhs4sg_rev_kernel \
  )
