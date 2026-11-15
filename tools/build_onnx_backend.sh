#!/usr/bin/env bash
# Build libtriton_onnxruntime.so and vendor ONNX Runtime.
#
# ONNX Runtime is taken from its official release archive (pinned by ORT_VERSION,
# and by ORT_SHA256 once known) and unpacked into third_party/triton-runtime/
# onnxruntime/. The backend is pointed at it through
# TRITON_ONNXRUNTIME_INCLUDE_PATHS/_LIB_PATHS: the alternative,
# TRITON_BUILD_ONNXRUNTIME_VERSION + TRITON_BUILD_CONTAINER, builds ORT inside a
# container image and is not usable here.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# the CMake-4 policy floor, the missing <cstdint> in the fetched common repo and RapidJSON
# live in one place so both builds agree
source "$(dirname "${BASH_SOURCE[0]}")/triton_build_env.sh"
backend="${root}/third_party/onnxruntime_backend"
prefix="${MADMOM_TRITON_RUNTIME:-${root}/third_party/triton-runtime}"
build="${root}/build/ort-backend"
downloads="${prefix}/downloads"
# Era-matched with the backend branch r24.08 (August 2024); the backend itself
# does not pin it, NVIDIA's CI does.
ORT_VERSION="${ORT_VERSION:-1.18.1}"
ORT_SHA256="${ORT_SHA256:-}"
ORT_URL="${ORT_URL:-https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz}"
# The backend pulls triton-inference-server/{common,core} through FetchContent; without
# these tags it takes `main`, which is a different generation than the core branch.
TRITON_TAG="${TRITON_TAG:-r24.08}"
# same CMake 4 policy floor as the core build (the fetched repositories declare 3.4)
POLICY_FLAG="${CMAKE_POLICY_FLAG:--DCMAKE_POLICY_VERSION_MINIMUM=3.5}"
# RapidJSON comes from triton_build_env.sh (RAPIDJSON_DIR + CMAKE_PREFIX_PATH)
if [ -z "${RAPIDJSON_DIR:-}" ]; then
  echo "warning: RapidJSON not found; the common repository will fail to configure" >&2
else
  echo "==> using RapidJSON from ${RAPIDJSON_DIR}"
fi
[ -f "${backend}/CMakeLists.txt" ] || {
  echo "error: ${backend} is empty; run: git submodule update --init --recursive" >&2; exit 1; }
branch="$(git -C "${backend}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
[ "${branch}" = "r24.08" ] || echo "warning: backend on '${branch}', expected r24.08 to match core" >&2
mkdir -p "${downloads}"
archive="${downloads}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
if [ ! -f "${archive}" ]; then
  echo "==> downloading ONNX Runtime ${ORT_VERSION}"
  curl -fL --retry 3 -o "${archive}.part" "${ORT_URL}"
  mv "${archive}.part" "${archive}"
fi
hash="$(sha256sum "${archive}" | cut -d' ' -f1)"
if [ -n "${ORT_SHA256}" ]; then
  [ "${hash}" = "${ORT_SHA256}" ] || {
    echo "error: ONNX Runtime checksum mismatch" >&2
    echo "  expected ${ORT_SHA256}" >&2; echo "  got      ${hash}" >&2; exit 1; }
else
  echo "note: ORT_SHA256 unset; computed ${hash} — pin it for reproducible builds"
fi
ort_dir="${prefix}/onnxruntime"
if [ ! -d "${ort_dir}/include" ]; then
  echo "==> unpacking ONNX Runtime into ${ort_dir}"
  rm -rf "${ort_dir}"; mkdir -p "${ort_dir}"
  tar -xzf "${archive}" -C "${ort_dir}" --strip-components=1
fi
echo "==> configuring the ONNX Runtime backend"
cmake -S "${backend}" -B "${build}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS_EXTRA:--include cstdint}" \
  -DCMAKE_CXX_FLAGS_RELEASE="${CMAKE_CXX_FLAGS_RELEASE_EXTRA:--O3 -DNDEBUG}" -DCMAKE_INSTALL_PREFIX="${prefix}" \
  -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_STATS=OFF \
  -DTRITON_ENABLE_ONNXRUNTIME_TENSORRT=OFF -DTRITON_ENABLE_ONNXRUNTIME_OPENVINO=OFF \
  -DTRITON_ONNXRUNTIME_INCLUDE_PATHS="${ort_dir}/include" \
  -DTRITON_ONNXRUNTIME_LIB_PATHS="${ort_dir}/lib" \
  -DTRITON_CORE_REPO_TAG="${TRITON_TAG}" -DTRITON_COMMON_REPO_TAG="${TRITON_TAG}" \
  "${POLICY_FLAG}" ${RAPIDJSON_DIR:+-DRapidJSON_DIR="${RAPIDJSON_DIR}"}
echo "==> building"
cmake --build "${build}" --parallel "$(nproc)"
cmake --install "${build}"
[ -f "${prefix}/backends/onnxruntime/libtriton_onnxruntime.so" ] || {
  echo "error: libtriton_onnxruntime.so not produced" >&2; exit 1; }
echo "==> done: ${prefix}/backends/onnxruntime/libtriton_onnxruntime.so"
