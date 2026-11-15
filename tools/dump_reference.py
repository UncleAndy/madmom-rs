#!/usr/bin/env python3
"""Dump intermediate reference stages of the flagship madmom pipeline.

The parity tests of this port compare single stages, so an error is caught at
the stage that introduced it instead of in the final detections. This script
runs the *reference* implementation (madmom, built in third_party/madmom) once
per input file and stores every stage as a .npy file.

Run it from the dev shell after building madmom's extensions:

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.cython])' --run \
      'python3 tools/dump_reference.py --audio third_party/madmom/tests/data/audio/sample.wav'

It requires madmom to be importable (built in place) and the model submodule to
be checked out; both failures are reported with the command that fixes them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

def add_madmom_to_path(madmom_path: Path) -> None:
    """Make the reference implementation importable from the submodule.

    madmom is not installed: the checkout is referenced in place, so its parent
    directory has to be on sys.path (``third_party/madmom`` contains the
    ``madmom`` package). The built extensions must live next to it, see
    tools/build_madmom_ext.py.
    """
    root = madmom_path.resolve()
    if not (root / "madmom" / "__init__.py").is_file():
        raise SystemExit(
            "%s does not look like a madmom checkout\n"
            "run: git submodule update --init --recursive" % root
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))



STAGES = ("signal", "frames", "stft", "spec", "filt", "logspec", "diff",
          "features", "activations", "path", "beats")


def import_madmom():
    """Import madmom or explain how to build it."""
    try:
        import madmom  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "cannot import madmom: {}\n"
            "build the reference implementation first:\n"
            "  nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.cython])' --run \\\n"
            "    'cd third_party/madmom && python3 setup.py build_ext --inplace'".format(exc)
        )


def build_stages(args):
    """Return an ordered list of (name, callable) producing the stage values."""
    import madmom
    from madmom.audio.signal import SignalProcessor, FramedSignalProcessor
    from madmom.audio.spectrogram import (FilteredSpectrogramProcessor,
                                         LogarithmicSpectrogramProcessor,
                                         SpectrogramProcessor,
                                         SpectrogramDifferenceProcessor)
    from madmom.audio.stft import ShortTimeFourierTransformProcessor
    from madmom.features.beats import DBNBeatTrackingProcessor

    sig = SignalProcessor(num_channels=1, sample_rate=args.sample_rate)
    frames = FramedSignalProcessor(frame_size=args.frame_size)
    stft = ShortTimeFourierTransformProcessor()
    spec = SpectrogramProcessor()
    filt = FilteredSpectrogramProcessor(num_bands=args.num_bands, fmin=args.fmin,
                                        fmax=args.fmax, norm_filters=True)
    logspec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    diff = SpectrogramDifferenceProcessor(diff_ratio=0.5, positive_diffs=True,
                                          stack_diffs=np.hstack)
    dbn = DBNBeatTrackingProcessor(fps=args.fps, min_bpm=args.min_bpm,
                                   max_bpm=args.max_bpm)

    stages = [
        ("signal", lambda _: sig(args.audio)),
        ("frames", lambda prev: frames(prev)),
        ("stft", lambda prev: stft(prev)),
        ("spec", lambda prev: spec(prev)),
        ("filt", lambda prev: filt(prev)),
        ("logspec", lambda prev: logspec(prev)),
        ("diff", lambda prev: diff(prev)),
    ]

    def nn_pair():
        """Return (network input, activations) of the public processor chain.

        The stage list above is a single-resolution chain (one band bank) — that
        is what the individual DSP stages are pinned against — but it is *not*
        what the networks receive: BEATS_BLSTM is fed 266 features per frame by a
        three-resolution chain, BEATS_LSTM 162 by a 12-band one. The NN stages
        therefore come from the public processor, exactly as tools/dump_preproc.py
        walks it, so the dump cannot disagree with the reference.
        """
        import dump_preproc
        data = args.audio          # the chain starts from the audio file
        for stage in getattr(dump_preproc.groups()[args.group](), "processors", []):
            if dump_preproc.is_network(stage):
                features = np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32)
                return features, np.asarray(stage(features))
            data = stage(data)
        raise SystemExit("no network stage found for group %r" % args.group)

    return stages, dbn, nn_pair


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, type=Path, help="input audio file")
    parser.add_argument("--madmom-path", type=Path, default=Path("third_party/madmom"),
                        help="checkout of the reference implementation (built in place)")
    parser.add_argument("--out", type=Path, default=Path("tests/golden"),
                        help="output directory for the stage dumps")
    parser.add_argument("--stages", nargs="+", default=list(STAGES),
                        help="stages to dump (default: all)")
    parser.add_argument("--frame-size", type=int, default=2048)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--group", default="beats_blstm",
                        help="processor group whose network stages are dumped")
    parser.add_argument("--num-bands", type=int, default=6)
    parser.add_argument("--fmin", type=float, default=30.0)
    parser.add_argument("--fmax", type=float, default=17000.0)
    parser.add_argument("--fps", type=float, default=100.0)
    parser.add_argument("--min-bpm", type=float, default=55.0)
    parser.add_argument("--max-bpm", type=float, default=215.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without touching madmom")
    args = parser.parse_args()
    add_madmom_to_path(args.madmom_path)

    if args.dry_run:
        print("would dump stages:", ", ".join(args.stages))
        print("input:", args.audio)
        print("output:", args.out)
        return 0

    import_madmom()
    args.out.mkdir(parents=True, exist_ok=True)
    stages, dbn, nn_pair = build_stages(args)

    previous = None
    for name, fn in stages:
        if name not in args.stages:
            continue
        previous = fn(previous) if previous is not None else fn(None)
        values = np.asarray(previous)
        path = args.out / f"{args.audio.stem}.{name}.npy"
        np.save(path, values)
        print(f"{name:12s} shape={values.shape} dtype={values.dtype} -> {path}")

    activations = None
    if {"features", "activations", "path", "beats"} & set(args.stages):
        features, activations = nn_pair()
        if "features" in args.stages:
            path = args.out / f"{args.audio.stem}.features.npy"
            np.save(path, features)
            print(f"{'features':12s} shape={features.shape} dtype={features.dtype} -> {path}")
        if "activations" in args.stages:
            path = args.out / f"{args.audio.stem}.activations.npy"
            np.save(path, np.asarray(activations))
            print(f"{'activations':12s} shape={np.shape(activations)} dtype={np.asarray(activations).dtype} -> {path}")

    if "path" in args.stages or "beats" in args.stages:
        activations = np.asarray(activations).flatten()
        path, log_prob = dbn.hmm.viterbi(activations)
        if "path" in args.stages:
            np.save(args.out / f"{args.audio.stem}.path.npy", path)
            print(f"{'path':12s} shape={path.shape} log_prob={log_prob:.6f}")
        if "beats" in args.stages:
            beats = dbn.process_offline(activations)
            np.save(args.out / f"{args.audio.stem}.beats.npy", beats)
            print(f"{'beats':12s} n={len(beats)} first={beats[:3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
