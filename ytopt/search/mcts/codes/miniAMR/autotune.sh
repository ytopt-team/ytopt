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
(cd "${SCRIPTPATH}" && python3 "${ROOTPATH}/mctree.py" --packing-arrays=allarray,allwork autotune --polybench-time --exec-args="--nx 94 --ny 94 --nz 94 --max_blocks 128 --uniform_refine 1 --num_refine 2" \
  --ld-library-path="${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  "${CLANG_PREFIX}/bin/clang" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native \
  *.c -o "${BASENAME}" \
  -I"/soft/libraries/mpi/openmpi/2.1.6/include" -L"/soft/libraries/mpi/openmpi/2.1.6/lib" -lmpi -lm -fcommon \
  -mllvm -polly-only-func=stencil7_calc_all \
  )
