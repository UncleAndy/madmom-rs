#!/usr/bin/env bash
# Shared environment for the Triton builds (sourced by build_triton_core.sh and
# build_onnx_backend.sh).
#
# What the build needs beyond the devShell, and why each item is here:
#
# 1. `<cstdint>` force-included. `triton-inference-server/common` declares
#    `enum class Code : uint8_t` in error.h without including <cstdint>, and GCC 13+ no
#    longer provides it transitively through <string>. One compiler flag fixes every
#    translation unit; the fetched sources stay untouched.
#
# 2. RapidJSON, numactl and Boost from nixpkgs. The core builds protobuf, Abseil, re2 and
#    googletest itself (its `third-party` repository), but these three it expects to find
#    installed — NVIDIA's images ship them. They are published through CMAKE_PREFIX_PATH
#    and the compiler flags, because the core configures its own sub-project through
#    ExternalProject, which forwards only the arguments it knows about but does inherit the
#    environment.
#
#    Deliberately *not* provided: protobuf and Abseil. nixpkgs has protobuf 34, whose C++
#    API (string_view field names, the JsonPrintOptions changes) does not compile against
#    Triton r24.08; the core's own pinned build is the right one.
#
# 3. `-Wno-error=maybe-uninitialized` in the per-configuration flags, because the ONNX
#    Runtime backend trips that warning on GCC 15 while the projects build with -Werror.
#    Per-configuration flags are used because they come *after* the projects' own -Werror.
set -u

# 1. missing <cstdint>
case " ${CXXFLAGS:-} " in
  *" -include cstdint "*) ;;
  *) export CXXFLAGS="${CXXFLAGS:-} -include cstdint" ;;
esac

# 2. system libraries
_extra_cxx=""
_extra_ld=""
for _spec in rapidjson numactl boost boost.dev; do
  command -v nix-build >/dev/null 2>&1 || break
  _path="$(nix-build '<nixpkgs>' -A "${_spec}" --no-out-link 2>/dev/null || true)"
  [ -n "${_path}" ] || continue
  _pkg="${_spec%%.*}"
  case ":${CMAKE_PREFIX_PATH:-}:" in
    *":${_path}:"*) ;;
    *) export CMAKE_PREFIX_PATH="${_path}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}" ;;
  esac
  if [ -d "${_path}/include" ]; then
    case "${_extra_cxx}" in
      *" -I${_path}/include"*) ;;
      *) _extra_cxx="${_extra_cxx} -I${_path}/include" ;;
    esac
  fi
  if [ -d "${_path}/lib" ]; then
    case "${_extra_ld}" in
      *" -L${_path}/lib"*) ;;
      *) _extra_ld="${_extra_ld} -L${_path}/lib" ;;
    esac
  fi
  echo "triton build env: ${_spec} ${_path}"
  # RapidJSON: triton-common includes <rapidjson/document.h> directly, so the package's
  # configuration directory has to be named for find_package as well
  if [ "${_pkg}" = rapidjson ] && [ -z "${RapidJSON_DIR:-}" ]; then
    for _candidate in "${_path}"/lib/cmake/RapidJSON "${_path}"/lib64/cmake/RapidJSON; do
      [ -d "${_candidate}" ] && export RapidJSON_DIR="${_candidate}" && break
    done
    unset _candidate
  fi
done
unset _pkg _spec _path

export CMAKE_CXX_FLAGS_EXTRA="${CMAKE_CXX_FLAGS_EXTRA:--include cstdint}${_extra_cxx}${_extra_ld}"
export CXXFLAGS="${CXXFLAGS:-} -include cstdint${_extra_cxx}${_extra_ld}"
export CFLAGS="${CFLAGS:-} -include stdint.h${_extra_cxx}${_extra_ld}"
export CMAKE_CXX_FLAGS_RELEASE_EXTRA="${CMAKE_CXX_FLAGS_RELEASE_EXTRA:--O3 -DNDEBUG -Wno-error=maybe-uninitialized}"
unset _extra_cxx _extra_ld
