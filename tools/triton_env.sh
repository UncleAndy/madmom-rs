#!/usr/bin/env bash
# Environment for the in-process Triton inference server.
#
# libtritonserver.so and the ONNX Runtime backend are built once, out of store,
# by tools/build_triton_core.sh and tools/build_onnx_backend.sh into
# third_party/triton-runtime/ (git-ignored). Source this before running
# madmom-cli or the triton-live tests:  source ./tools/triton_env.sh
set -u
_madmom_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_runtime="${MADMOM_TRITON_RUNTIME:-${_madmom_root}/third_party/triton-runtime}"
export MADMOM_TRITON_LIB="${_runtime}/lib"
export MADMOM_TRITON_BACKENDS="${_runtime}/backends"
export MADMOM_TRITON_ORT="$(ls -d "${_runtime}"/onnxruntime*/lib 2>/dev/null | head -1)"
export MADMOM_MODEL_REPO="${MADMOM_MODEL_REPO:-${_madmom_root}/third_party/madmom-models-onnx/models}"
for _dir in "${MADMOM_TRITON_ORT:-}" "${MADMOM_TRITON_LIB}" "${MADMOM_TRITON_BACKENDS}"; do
  [ -n "${_dir}" ] || continue
  if [ -d "${_dir}" ]; then
    export LD_LIBRARY_PATH="${_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LIBRARY_PATH="${_dir}${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  else
    echo "triton_env.sh: warning: ${_dir} missing (run tools/build_triton_core.sh)" >&2
  fi
done
[ -d "${MADMOM_MODEL_REPO}" ] || echo "triton_env.sh: warning: no model repository at ${MADMOM_MODEL_REPO}" >&2
unset _madmom_root _runtime _dir
