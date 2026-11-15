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
# the CMake-4 policy floor, the missing <cstdint> in the fetched common repo and RapidJSON
# live in one place so both builds agree
source "$(dirname "${BASH_SOURCE[0]}")/triton_build_env.sh"
core="${root}/third_party/tritonserver-rs/core"
prefix="${MADMOM_TRITON_RUNTIME:-${root}/third_party/triton-runtime}"
build="${root}/build/triton-core"
triton_tag="${TRITON_TAG:-r24.08}"
# CMake 4 removed compatibility with `cmake_minimum_required(VERSION <3.5)`, and several
# of the third-party repositories fetched at r24.08 (pybind11 first) still declare 3.4.
# The policy floor tells CMake to configure them with 3.5 semantics instead of failing.
policy_flag="${CMAKE_POLICY_FLAG:--DCMAKE_POLICY_VERSION_MINIMUM=3.5}"
[ -f "${core}/include/triton/core/tritonserver.h" ] || {
  echo "error: ${core} is empty; run: git submodule update --init --recursive" >&2; exit 1; }
for tool in cmake gcc g++ git; do
  command -v "${tool}" >/dev/null || { echo "error: ${tool} not found (use: nix develop)" >&2; exit 1; }
done
echo "==> configuring Triton core (${triton_tag})"
# CMAKE_INSTALL_LIBDIR is pinned to "lib" on purpose: the core decides where its own
# third-party installs land from a hardcoded LIB_DIR that is "lib" everywhere except
# CentOS, while GNUInstallDirs picks "lib64" on this platform. Without the pin the two
# disagree and the core cannot find the protobuf, re2 and abseil it has just built.
cmake -S "${core}" -B "${build}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS_EXTRA:--include cstdint}" \
  -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE_EXTRA:--O3 -DNDEBUG}" -DCMAKE_INSTALL_PREFIX="${prefix}" \
  "${policy_flag}" \
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
