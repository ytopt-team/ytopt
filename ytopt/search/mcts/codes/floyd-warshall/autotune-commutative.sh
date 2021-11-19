#! /usr/bin/env bash
set -e

SCRIPTPATH=`realpath --no-symlinks $(dirname $0)`
BASENAME=`basename "${SCRIPTPATH}"`
ROOTPATH=`realpath --no-symlinks "${SCRIPTPATH}/../.."`

if [[ -z "${CLANG_PREFIX}" ]]; then
  echo "CLANG_PREFIX envirnment variable not set"
  exit 1
fi

(cd "${SCRIPTPATH}" && python3 "${ROOTPATH}/mctree.py" --no-threading autotune --ld-library-path="${CLANG_PREFIX}/lib" \
  "${CLANG_PREFIX}/bin/clang" -I"${CLANG_PREFIX}/projects/openmp/runtime/src" -I"${CLANG_PREFIX}/runtimes/runtimes-bins/openmp/runtime/src" \
  -flegacy-pass-manager -mllvm -polly-position=early -O3 -march=native \
  *.c -o "${BASENAME}" \
  -DLARGE_DATASET -I"${SCRIPTPATH}" -mllvm -polly-pragma-ignore-depcheck=1 \
  -mllvm -polly-only-func=kernel_floyd_warshall \
  )
