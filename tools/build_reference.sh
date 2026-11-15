#!/usr/bin/env bash
# Build the reference implementation (madmom) in place and record the test
# baseline. This baseline is the honest starting point of the port: whatever
# fails here (missing opencv/pyfftw/pyaudio, NumPy differences) is recorded, not
# "fixed", so that later we can tell our own regressions from pre-existing ones.
#
# Result: third_party/madmom/*.so (git-ignored build products inside the
# submodule) and tools/reference-baseline.txt.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
madmom="${root}/third_party/madmom"
out="${root}/tools/reference-baseline.txt"
PY_PACKAGES='python3.withPackages(ps: [ps.numpy ps.scipy ps.cython ps.pytest ps.packaging ps.setuptools])'

[ -f "${madmom}/setup.py" ] || {
  echo "error: ${madmom} is empty; run: git submodule update --init --recursive" >&2; exit 1; }
[ -d "${madmom}/madmom/models" ] || {
  echo "error: ${madmom}/madmom/models is empty (nested submodule);" >&2
  echo "       run: git submodule update --init --recursive" >&2; exit 1; }

echo "==> building madmom's Cython/C extensions in place"
nix-shell -p "${PY_PACKAGES}" --run "cd '${madmom}' && python3 setup.py build_ext --inplace"

echo "==> running the reference test suite (baseline)"
set +e
nix-shell -p "${PY_PACKAGES}" --run \
  "cd '${madmom}' && python3 -m pytest tests -q -p no:cacheprovider" 2>&1 | tee "${out}"
status=${PIPESTATUS[0]}
set -e
echo "==> pytest exit status ${status}; baseline written to ${out}"
echo "    (non-zero is expected while optional dependencies are missing)"
