#!/usr/bin/env python3
"""Dump the reference tempo estimation stages.

Walks madmom's own public processors over the reference activation fixtures, so every
stage the port has to reproduce is pinned without the network:

* `CombFilterTempoHistogramProcessor(fps=100)` and `TempoEstimationProcessor(fps=100)`
  on `sample.beats_blstm.npz` (the beat activation function, 281 frames at 100 fps);
* the same with `ACFTempoHistogramProcessor` and `DBNTempoHistogramProcessor`;
* `TCNTempoHistogramProcessor(min_bpm=10, max_bpm=250)` on the multi-task TCN output
  `sample.beats_tcn_tempo.npz`;
* the plain functions (`interval_histogram_{acf,comb}`, `smooth_histogram`,
  `dominant_interval`, `detect_tempo`).

Writes into `tests/golden/tempo/` plus an `index.tsv` with tag, shape and sha256.

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' --run \\
      'python3 tools/dump_tempo.py third_party/madmom'
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

MIN_TAU = 24
MAX_TAU = 150


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root = Path(argv[1]).resolve()
    sys.path.insert(0, str(root))
    from madmom.features import Activations
    from madmom.features.tempo import (
        ACFTempoHistogramProcessor,
        CombFilterTempoHistogramProcessor,
        DBNTempoHistogramProcessor,
        TCNTempoHistogramProcessor,
        TempoEstimationProcessor,
        detect_tempo,
        dominant_interval,
        interval_histogram_acf,
        interval_histogram_comb,
        smooth_histogram,
    )
    from madmom.audio.signal import smooth as smooth_signal

    act_dir = root / "tests" / "data" / "activations"
    with np.load(act_dir / "sample.beats_blstm.npz") as data:
        act = np.asarray(data["activations"], dtype=float)
        fps = float(np.asarray(data["fps"]))
    print("beat activations", act.shape, "fps", fps)

    out = Path("tests/golden/tempo")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(tag, values):
        values = np.asarray(values)
        if values.dtype == np.float64 or values.dtype == np.int64:
            pass
        target = out / ("%s.npy" % tag)
        np.save(target, values)
        shape = values.shape or (1,)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append("%s\t%s\t%s\n" % (tag, "x".join(str(d) for d in shape), digest))
        print("%-22s %-10s %s" % (tag, values.shape, digest[:12]))

    # ---- plain functions -------------------------------------------------------
    save("act", act)
    hist = interval_histogram_comb(act, 0.79, MIN_TAU, MAX_TAU)
    save("hist_comb_bins", hist[0])
    save("hist_comb_delays", hist[1])
    acf = interval_histogram_acf(act, MIN_TAU, MAX_TAU)
    save("hist_acf_bins", acf[0])
    save("hist_acf_delays", acf[1])
    save("hist_comb_bins_smooth3", smooth_histogram(hist, 3)[0])
    save("dominant_interval", np.array([dominant_interval(hist)]))
    save("detect_tempo_hist_bpm", detect_tempo(hist, fps))
    save("detect_tempo_hist_delays", detect_tempo(hist))
    save("detect_tempo_hist_interp", detect_tempo(hist, fps, interpolate=True))
    save("act_smoothed", smooth_signal(act, int(round(fps * 0.14))))

    # ---- comb filter histogram processor ---------------------------------------
    comb = CombFilterTempoHistogramProcessor(fps=fps)
    comb_hist = comb(act)
    save("comb_hist_bins", comb_hist[0])
    save("comb_hist_delays", comb_hist[1])
    save("comb_hist_bins_smooth9", smooth_histogram(comb_hist, 9)[0])
    save("tempi_comb", TempoEstimationProcessor(fps=fps)(act))
    save("tempi_comb_interp", TempoEstimationProcessor(fps=fps, interpolate=True)(act))

    # ---- acf histogram processor ----------------------------------------------
    acf_proc = ACFTempoHistogramProcessor(fps=fps)
    acf_hist = acf_proc(act)
    save("acf_hist_bins", acf_hist[0])
    save("acf_hist_delays", acf_hist[1])
    save("tempi_acf", TempoEstimationProcessor(histogram_processor=acf_proc, fps=fps)(act))

    # ---- dbn histogram processor ----------------------------------------------
    dbn_proc = DBNTempoHistogramProcessor(fps=fps)
    dbn_hist = dbn_proc(act)
    save("dbn_hist_bins", dbn_hist[0])
    save("dbn_hist_delays", dbn_hist[1])
    save("tempi_dbn", TempoEstimationProcessor(histogram_processor=dbn_proc, fps=fps,
                                               act_smooth=None)(act))

    # ---- tcn tempo histogram ---------------------------------------------------
    tcn_act = Activations(str(act_dir / "sample.beats_tcn_tempo.npz"))
    tcn_proc = TCNTempoHistogramProcessor(min_bpm=10, max_bpm=250)
    tcn_hist = tcn_proc(tcn_act)
    save("tcn_hist_bins", tcn_hist[0])
    save("tcn_hist_delays", tcn_hist[1])
    save("tcn_tempi", TempoEstimationProcessor(histogram_processor=tcn_proc,
                                               act_smooth=None)(tcn_act))
    save("tcn_tempi_smooth15", TempoEstimationProcessor(histogram_processor=tcn_proc,
                                                       act_smooth=None, hist_smooth=15)(tcn_act))

    (out / "index.tsv").write_text("".join(rows), encoding="utf-8", newline="\n")
    print("wrote", out / "index.tsv", "with", len(rows), "cases")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
