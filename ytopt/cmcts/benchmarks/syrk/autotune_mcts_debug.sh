#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`
SAVEPATH="/gpfs/jlse-fs0/users/jkoo/exp"
if [ -d "${SAVEPATH}/${BASENAME}_mcts" ] 
then
    echo "Directory exists." 
else
    echo "Error: Directory does not exists."
    mkdir "${SAVEPATH}/${BASENAME}_mcts"
fi

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX environment variable not set"
  exit 1
fi

(cd "${SCRIPTPATH}" && python3 "${ROOTPATH}/mctree_mcts_debug.py" --packing-arrays=C,A,B autotune --outdir="${SAVEPATH}/${BASENAME}_mcts" --polybench-time \
  --ld-library-path="${CLANG_PREFIX}/lib:${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  "${CLANG_PREFIX}/bin/clang" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" -L"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native \
  *.c -o "${BASENAME}" \
  -I"${SCRIPTPATH}" \
  -DLARGE_DATASET -mllvm -polly-only-func=kernel_syrk \
  )
