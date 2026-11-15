#!/usr/bin/env python3
"""Convert madmom's pickled models into an ONNX model repository for Triton.

Layout produced (see plan section 2.1):

    <dst>/<model_name>/config.pbtxt
    <dst>/<model_name>/1/model.onnx

Model names are the pickle stems (`beats_lstm_1`, `ballroom_pattern_4_4`, ...),
ensembles stay eight separate models. Families without a builder yet (CNN, TCN,
CRF, GMM) are reported as `unsupported` instead of being silently skipped: the
report is the acceptance list for plan task 2.4.

Example:

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.onnx])' --run \\
      'python3 tools/convert_to_onnx.py \\
         --src third_party/madmom/madmom/models \\
         --dst third_party/madmom-models-onnx/models --report models-report.json'
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import onnx_spec  # noqa: E402  (local module, same directory)

# Families that must be converted for the first release, with the phase that
# brings the missing builders (plan tasks 2.x, 4.5, 4.6).
REQUIRED_FAMILIES = ("lstm", "blstm", "gru", "bgru", "rnn", "brnn", "feedforward",
                     "cnn", "tcn", "crf", "gmm")


def iter_pickles(src: Path) -> list[Path]:
    """All model pickles under `src`, in a stable order."""
    return sorted(p for p in src.rglob("*.pkl"))


def sha256_of(path: Path) -> str:
    """Checksum of a file, used to pin the produced repository."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_one(pickle_path: Path, dst: Path, *, validate: bool, dry_run: bool) -> dict:
    """Convert a single pickle, returning a report row."""
    row: dict = {"pickle": str(pickle_path), "name": pickle_path.stem}
    try:
        model = onnx_spec.load_model(pickle_path)
    except Exception as exc:  # load failures are report rows, not crashes
        row.update(status="load-error", error="%s: %s" % (type(exc).__name__, exc))
        return row
    family = onnx_spec.model_family(model)
    spec = onnx_spec.spec_for(pickle_path, family=family)
    row.update(family=family, layers=len(onnx_spec.layers_of(model)))
    if dry_run:
        row.update(status="planned", config=spec.config_pbtxt())
        return row
    try:
        onnx_model = onnx_spec.build_model(model, spec)
    except onnx_spec.UnsupportedLayer as exc:
        row.update(status="unsupported", error=str(exc))
        return row
    except SystemExit as exc:
        row.update(status="no-onnx-package", error=str(exc))
        return row
    if validate:
        onnx, _, _ = onnx_spec._require_onnx()
        onnx.checker.check_model(onnx_model)
        row["validated"] = True
    model_dir = dst / spec.name / "1"
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / "model.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())
    (dst / spec.name / "config.pbtxt").write_text(spec.config_pbtxt(), encoding="utf-8")
    row.update(status="converted", path=str(onnx_path), sha256=sha256_of(onnx_path),
               config=spec.config_pbtxt())
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True,
                        help="madmom model directory (third_party/madmom/madmom/models)")
    parser.add_argument("--dst", type=Path, required=True,
                        help="output ONNX repository (third_party/madmom-models-onnx/models)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="convert only these model names (stems)")
    parser.add_argument("--report", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--validate", action="store_true",
                        help="run onnx.checker on every produced model")
    parser.add_argument("--dry-run", action="store_true",
                        help="report families and configs without writing anything")
    args = parser.parse_args()

    if not args.src.is_dir():
        raise SystemExit("%s does not exist; run: git submodule update --init --recursive" % args.src)

    pickles = iter_pickles(args.src)
    if args.only:
        wanted = set(args.only)
        pickles = [p for p in pickles if p.stem in wanted]
    if not pickles:
        raise SystemExit("no model pickles found in %s" % args.src)

    rows = [convert_one(p, args.dst, validate=args.validate, dry_run=args.dry_run)
            for p in pickles]

    by_status: dict[str, int] = {}
    by_family: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        by_family[row.get("family", "?")] = by_family.get(row.get("family", "?"), 0) + 1

    for row in rows:
        line = "%-28s %-12s %-12s" % (row["name"], row.get("family", "?"), row["status"])
        if row["status"] != "converted":
            line += " %s" % row.get("error", "")
        print(line)
    print("\nstatus:", ", ".join("%s=%d" % item for item in sorted(by_status.items())))
    print("families:", ", ".join("%s=%d" % item for item in sorted(by_family.items())))
    missing = [fam for fam in REQUIRED_FAMILIES if fam not in by_family]
    if missing:
        print("families without any model in this run:", ", ".join(missing))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"rows": rows, "status": by_status,
                                           "families": by_family}, indent=2),
                               encoding="utf-8")
        print("report:", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
