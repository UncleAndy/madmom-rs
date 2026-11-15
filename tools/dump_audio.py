#!/usr/bin/env python3
"""Dump madmom's decoded audio for the reference fixtures.

Writes one `.npy` per case into `tests/golden/io/` (in the file's *native* dtype, so a
comparison also proves which format we read) plus `tests/golden/io/index.tsv`:

    fixture<TAB>start<TAB>stop<TAB>sample_rate<TAB>descr<TAB>shape<TAB>sha256

`crates/madmom-io/tests/audio_golden.rs` replays the same cases through the Rust WAV
reader and requires bit-exact samples. The dumps are derived data and stay out of the
repository (see tests/golden/README.md).

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' --run \\
      'python3 tools/dump_audio.py third_party/madmom'
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

CASES = (
    # (fixture, start, stop)
    ("sample.wav", None, None),
    ("sample_22050.wav", None, None),
    ("sample2.wav", None, None),
    ("stereo_sample.wav", None, None),
    ("stereo_chirp.wav", None, None),
    ("sample.wav", 0.1, None),
    ("sample.wav", 0.1, 0.5),
    ("sample.wav", None, 0.25),
    ("stereo_sample.wav", 0.2, 0.6),
)


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root = Path(argv[1]).resolve()
    sys.path.insert(0, str(root))
    from madmom.io.audio import load_wave_file

    audio_dir = root / "tests" / "data" / "audio"
    out = Path("tests/golden/io")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fixture, start, stop in CASES:
        path = audio_dir / fixture
        signal, sample_rate = load_wave_file(str(path), start=start, stop=stop)
        tag = "%s%s%s" % (
            fixture,
            "" if start is None else ".start%s" % start,
            "" if stop is None else ".stop%s" % stop,
        )
        target = out / ("%s.npy" % tag)
        np.save(target, signal)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append("%s\t%s\t%s\t%d\t%s\t%s\t%s\n" % (
            fixture,
            "" if start is None else start,
            "" if stop is None else stop,
            sample_rate,
            signal.dtype.str,
            "x".join(str(dim) for dim in signal.shape),
            digest,
        ))
        print("%-42s %-8s %-10s %d Hz sha256=%s" % (tag, signal.dtype, signal.shape,
                                                    sample_rate, digest[:12]))
    (out / "index.tsv").write_text("".join(rows), encoding="utf-8", newline="\n")
    print("wrote", out / "index.tsv", "with", len(rows), "cases")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
