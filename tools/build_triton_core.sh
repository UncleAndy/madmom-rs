#!/usr/bin/env bash
# Build libtritonserver.so from third_party/tritonserver-rs/core (CPU-only).
#
# The core pulls and builds its third-party dependencies (gRPC, protobuf, absl,
# re2, prometheus-cpp, ...) from triton-inference-server/third_party at configure
# time: this needs network access and takes tens of minutes. The tag of those
# repositories must match the core branch, otherwise the build disagrees with
# itself.
#
# Result: third_party/triton-runtime/lib/libtritonserver.so (git-ignored).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
core="${root}/third_party/tritonserver-rs/core"
prefix="${MADMOM_TRITON_RUNTIME:-${root}/third_party/triton-runtime}"
build="${root}/build/triton-core"
triton_tag="${TRITON_TAG:-r24.08}"
[ -f "${core}/include/triton/core/tritonserver.h" ] || {
  echo "error: ${core} is empty; run: git submodule update --init --recursive" >&2; exit 1; }
for tool in cmake gcc g++ git; do
  command -v "${tool}" >/dev/null || { echo "error: ${tool} not found (use: nix develop)" >&2; exit 1; }
done
echo "==> configuring Triton core (${triton_tag})"
cmake -S "${core}" -B "${build}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${prefix}" \
  -DTRITON_CORE_HEADERS_ONLY=OFF \
  -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_MALI_GPU=OFF \
  -DTRITON_ENABLE_METRICS=OFF -DTRITON_ENABLE_METRICS_GPU=OFF -DTRITON_ENABLE_METRICS_CPU=OFF \
  -DTRITON_ENABLE_STATS=OFF -DTRITON_ENABLE_TRACING=OFF -DTRITON_ENABLE_NVTX=OFF \
  -DTRITON_ENABLE_ENSEMBLE=OFF -DTRITON_ENABLE_GCS=OFF -DTRITON_ENABLE_S3=OFF \
  -DTRITON_ENABLE_AZURE_STORAGE=OFF \
  -DTRITON_COMMON_REPO_TAG="${triton_tag}" -DTRITON_THIRD_PARTY_REPO_TAG="${triton_tag}"
echo "==> building (long)"
cmake --build "${build}" --parallel "$(nproc)"
echo "==> installing into ${prefix}"
cmake --install "${build}"
[ -f "${prefix}/lib/libtritonserver.so" ] || {
  echo "error: ${prefix}/lib/libtritonserver.so not produced" >&2
  echo "hint: TRITON_CORE_HEADERS_ONLY=OFF must be honoured (headers-only is the default)" >&2
  exit 1; }
echo "==> done: ${prefix}/lib/libtritonserver.so"
