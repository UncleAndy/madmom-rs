#!/usr/bin/env python3
"""Dump the reference onset detection stages.

Two independent groups, both taken from madmom itself:

* the eleven onset detection functions, evaluated on the spectrogram the reference's own
  tests build — `Spectrogram(sample.wav, circular_shift=True)` and its logarithmic filtered
  version (24 bands, `mul=1`, `add=1`) — together with that spectrogram, its filtered
  version and the phase, so the port's input can be checked before the ODFs are;
* the peak picker and the `OnsetPeakPickingProcessor`, evaluated on madmom's activation
  fixtures (`sample.super_flux.npz`, `sample.onsets_rnn.npz`), including the online mode.

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' --run \\
      'python3 tools/dump_onsets.py third_party/madmom'
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

NUM_BANDS = 24


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root = Path(argv[1]).resolve()
    sys.path.insert(0, str(root))
    from madmom.audio.spectrogram import (
        LogarithmicFilteredSpectrogram,
        Spectrogram,
    )
    from madmom.features import Activations
    from madmom.features.onsets import (
        OnsetPeakPickingProcessor,
        complex_domain,
        complex_flux,
        high_frequency_content,
        modified_kullback_leibler,
        normalized_weighted_phase_deviation,
        peak_picking,
        phase_deviation,
        rectified_complex_domain,
        spectral_diff,
        spectral_flux,
        superflux,
        weighted_phase_deviation,
    )

    audio = root / "tests" / "data" / "audio" / "sample.wav"
    spectrogram = Spectrogram(str(audio), circular_shift=True)
    log_filt_spec = LogarithmicFilteredSpectrogram(
        spectrogram, num_bands=NUM_BANDS, mul=1, add=1
    )
    phase = spectrogram.stft.phase()
    print("spectrogram", spectrogram.shape, "filtered", log_filt_spec.shape,
          "phase", phase.shape)

    out = Path("tests/golden/onsets")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(tag, values):
        values = np.asarray(values)
        target = out / ("%s.npy" % tag)
        np.save(target, values)
        shape = values.shape or (1,)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append("%s\t%s\t%s\n" % (tag, "x".join(str(d) for d in shape), digest))
        print("%-28s %-12s %s" % (tag, values.shape, digest[:12]))

    # ---- inputs ---------------------------------------------------------------
    save("spec", spectrogram)
    save("log_filt_spec", log_filt_spec)
    save("phase", phase)

    # ---- onset detection functions -------------------------------------------
    for tag, values in [
        ("hfc", high_frequency_content(log_filt_spec)),
        ("spectral_diff", spectral_diff(log_filt_spec)),
        ("spectral_flux", spectral_flux(log_filt_spec)),
        ("superflux", superflux(log_filt_spec)),
        ("complex_flux", complex_flux(log_filt_spec)),
        ("mkl", modified_kullback_leibler(log_filt_spec)),
        ("phase_deviation", phase_deviation(log_filt_spec)),
        # the phase-weighted ODFs refuse a filtered spectrogram (the phase has the bins of
        # the unfiltered one), so the reference's tests use the raw spectrogram for them
        ("weighted_phase_deviation", weighted_phase_deviation(spectrogram)),
        ("normalized_wpd", normalized_weighted_phase_deviation(spectrogram)),
        ("complex_domain", complex_domain(spectrogram)),
        ("rectified_complex_domain", rectified_complex_domain(spectrogram)),
    ]:
        save("odf_" + tag, values)

    # ---- peak picking --------------------------------------------------------
    act_dir = root / "tests" / "data" / "activations"
    superflux = np.asarray(Activations(str(act_dir / "sample.super_flux.npz")),
                           dtype=float)
    rnn = np.asarray(Activations(str(act_dir / "sample.onsets_rnn.npz")), dtype=float)
    print("superflux activations", superflux.shape, "rnn activations", rnn.shape)

    save("pp_superflux_default", peak_picking(superflux, 1.1))
    save("pp_superflux_smooth3", peak_picking(superflux, 1.1, smooth=3))
    save("pp_superflux_params",
         peak_picking(superflux, 1.1, pre_max=2, post_max=10, pre_avg=30))
    save("pp_rnn_post_max0", peak_picking(rnn, threshold=0.23, post_max=0))

    fps = float(np.asarray(Activations(str(act_dir / "sample.super_flux.npz")).fps))
    processor = OnsetPeakPickingProcessor(
        threshold=1.1, pre_max=0.01, post_max=0.05, pre_avg=0.15, post_avg=0,
        combine=0.03, delay=0, fps=fps)
    save("onsets_offline", processor(superflux))
    rnn_fps = float(np.asarray(Activations(str(act_dir / "sample.onsets_rnn.npz")).fps))
    online = OnsetPeakPickingProcessor(threshold=0.23, online=True, fps=rnn_fps)
    save("onsets_online_all", online(rnn))
    online.reset()
    frame_by_frame = np.hstack([online(np.atleast_1d(f), reset=False) for f in rnn])
    save("onsets_online_frames", frame_by_frame)

    (out / "index.tsv").write_text("".join(rows), encoding="utf-8", newline="\n")
    print("wrote", out / "index.tsv", "with", len(rows), "cases")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
