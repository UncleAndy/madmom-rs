#!/usr/bin/env python3
"""Dump madmom's filterbanks and frequency tables as reference dumps.

Writes `tests/golden/filters/<tag>.npy` plus `index.tsv` (`tag`, `shape`, `sha256`).
The Rust side (`crates/madmom-audio/tests/filters_golden.rs`) constructs the very same
cases from the tag names, so a renamed case fails loudly instead of being skipped.

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' --run \\
      'python3 tools/dump_filters.py third_party/madmom'
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    sys.path.insert(0, str(Path(argv[1]).resolve()))
    from madmom.audio.filters import (BarkFilterbank, LogarithmicFilterbank, MelFilterbank,
                                      bark_double_frequencies, bark_frequencies, erb2hz, hz2erb,
                                      hz2mel, hz2midi, log_frequencies, mel_frequencies,
                                      midi2hz, semitone_frequencies)

    # the bin frequencies the flagship chains use: 2048-point FFT at 44.1 kHz
    bin_frequencies = np.fft.fftfreq(2048, 1.0 / 44100.0)[:1024]
    out_dir = Path("tests/golden/filters")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = {
        "filterbank_log12_unique": lambda: LogarithmicFilterbank(bin_frequencies, 12, 30., 17000., 440., True, True),
        "filterbank_log12_all": lambda: LogarithmicFilterbank(bin_frequencies, 12, 30., 17000., 440., True, False),
        "filterbank_log3_unique": lambda: LogarithmicFilterbank(bin_frequencies, 3, 30., 17000., 440., True, True),
        "filterbank_log6_unique": lambda: LogarithmicFilterbank(bin_frequencies, 6, 30., 17000., 440., True, True),
        "filterbank_mel40_unit": lambda: MelFilterbank(bin_frequencies, 40, 20., 17000., True, True),
        "filterbank_mel40_all": lambda: MelFilterbank(bin_frequencies, 40, 20., 17000., True, False),
        "filterbank_bark_normal": lambda: BarkFilterbank(bin_frequencies, num_bands='normal', fmin=20., fmax=15500., norm_filters=True, unique_filters=True),
        "filterbank_bark_double": lambda: BarkFilterbank(bin_frequencies, num_bands='double', fmin=20., fmax=15500., norm_filters=True, unique_filters=True),
        "conversions_mel_frequencies_3": lambda: mel_frequencies(3, 20., 17000.),
        "conversions_mel_frequencies_40": lambda: mel_frequencies(40, 20., 17000.),
        "conversions_log_frequencies_12": lambda: log_frequencies(12, 30., 17000., 440.),
        "conversions_log_frequencies_3": lambda: log_frequencies(3, 30., 17000., 440.),
        "conversions_semitone_400_500": lambda: semitone_frequencies(400., 500., 440.),
        "conversions_bark_normal": lambda: bark_frequencies(20., 15500.),
        "conversions_bark_double": lambda: bark_double_frequencies(20., 15500.),
        "conversions_mel_grid": lambda: hz2mel(np.array([30., 100., 440., 1000., 17000.])),
        "conversions_erb_grid": lambda: hz2erb(np.array([30., 100., 440., 1000., 17000.])),
        "conversions_midi_grid": lambda: hz2midi(np.array([30., 100., 440., 1000., 17000.]), 440.),
        "conversions_midi_to_hz": lambda: midi2hz(np.array([21., 40., 69., 100., 127.]), 440.),
        "conversions_erb_to_hz": lambda: erb2hz(np.array([1., 5., 10., 20.])),
    }

    rows = []
    for tag, build in cases.items():
        values = np.asarray(build())
        if values.dtype == np.float64:
            values = values.astype(np.float32)
        target = out_dir / ("%s.npy" % tag)
        np.save(target, values)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        shape = values.shape if values.ndim else (int(values.size),)
        rows.append("%s\t%s\t%s\n" % (tag, "x".join(str(dim) for dim in shape), digest))
        print("%-30s %-14s %s" % (tag, values.dtype, shape))
    (out_dir / "index.tsv").write_text("".join(rows), encoding="utf-8", newline="\n")
    print("wrote", out_dir / "index.tsv", "with", len(rows), "cases")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
