#!/usr/bin/env python3
"""Compare an ONNX model against the reference network it was converted from.

The reference side is produced by tools/dump_preproc.py: it writes the exact
input each network of an ensemble receives (`preproc/<group>.npy`) and that
network's own output (`activations/<group>_<n>.npy`). This script runs the
converted ONNX model on the former and compares against the latter, so a
conversion mistake (gate order, transposed weights, wrong activation) shows up
as a number instead of as a mysterious downstream drift.

Tolerances are the ones fixed in the plan (§4.2): |a-b| <= 1e-4 for network
activations, cosine similarity >= 0.999.

Usage:
  python3 tools/validate_onnx.py --model-repo /tmp/onnx-repo \\
      --golden tests/golden --group beats_lstm --network 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ATOL = 1e-4
MIN_COSINE = 0.999


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-repo", type=Path, required=True,
                        help="ONNX repository produced by tools/convert_to_onnx.py")
    parser.add_argument("--golden", type=Path, default=Path("tests/golden"),
                        help="directory holding preproc/ and activations/ dumps")
    parser.add_argument("--group", required=True, help="processor group, e.g. beats_lstm")
    parser.add_argument("--network", type=int, default=1,
                        help="1-based network index inside the group (ensembles have several)")
    parser.add_argument("--model", default=None,
                        help="model name in the repository (default: <group>_<network>)")
    args = parser.parse_args()

    preproc = args.golden / "preproc" / ("%s.npy" % args.group)
    reference = args.golden / "activations" / ("%s_%d.npy" % (args.group, args.network))
    for path in (preproc, reference):
        if not path.is_file():
            raise SystemExit("%s is missing; run tools/dump_preproc.py first" % path)

    import onnxruntime as ort

    name = args.model or "%s_%d" % (args.group, args.network)
    onnx_path = args.model_repo / name / "1" / "model.onnx"
    if not onnx_path.is_file():
        raise SystemExit("%s is missing; run tools/convert_to_onnx.py first" % onnx_path)

    features = np.ascontiguousarray(np.load(preproc), dtype=np.float32)
    expected = np.asarray(np.load(reference))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (input_meta,) = session.get_inputs()
    (output_meta,) = session.get_outputs()

    feed = features
    if len(input_meta.shape) == 3 and features.ndim == 2:
        feed = features[np.newaxis, ...]
    elif len(input_meta.shape) == 2 and features.ndim == 3 and features.shape[0] == 1:
        feed = features[0]

    got = np.asarray(session.run([output_meta.name], {input_meta.name: feed})[0])

    if expected.shape != got.shape:
        raise SystemExit("shape mismatch: reference %s vs onnx %s" % (expected.shape, got.shape))

    a = expected.ravel().astype(np.float64)
    b = got.ravel().astype(np.float64)
    absdiff = np.abs(a - b)
    cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if a.any() and b.any() else 1.0
    ok = absdiff.max() <= ATOL and cosine >= MIN_COSINE

    print("%s (%s): input %s -> onnx %s, reference %s" %
          (name, input_meta.name, tuple(feed.shape), got.shape, expected.shape))
    print("  max abs diff %.3e (atol %.0e), mean %.3e, cosine %.8f (min %.3f)" %
          (absdiff.max(), ATOL, absdiff.mean(), cosine, MIN_COSINE))
    print("  %s" % ("MATCH" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
