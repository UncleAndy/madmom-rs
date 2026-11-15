#!/usr/bin/env python3
"""Dump the reference beat trackers.

Everything runs on madmom's own activation fixtures (`sample.beats_blstm.npz` for the
offline trackers, `sample.beats_lstm.npz` for the online/forward one), so the networks are
replaced by their outputs and the trackers are what is pinned:

* `DBNBeatTrackingProcessor` — the default path, `correct=False`, the two thresholds, the
  synthetic activation with beats on the first and last frame, the empty path, and the
  forward (online) mode both all-at-once and frame by frame, with and without a reset;
* `BeatTrackingProcessor` (local tempo) and `BeatDetectionProcessor` (global tempo);
* `CRFBeatDetectionProcessor`, including the candidate intervals it tries;
* the `detections/*.txt` files as they were produced by the command line programs, as an
  end-to-end check that the fixture activations are the ones those files came from.

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' --run \\
      'python3 tools/dump_beats.py third_party/madmom'
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root = Path(argv[1]).resolve()
    sys.path.insert(0, str(root))
    from madmom.features import Activations
    from madmom.features.beats import (
        BeatDetectionProcessor,
        BeatTrackingProcessor,
        CRFBeatDetectionProcessor,
        DBNBeatTrackingProcessor,
    )

    act_dir = root / "tests" / "data" / "activations"
    blstm = Activations(str(act_dir / "sample.beats_blstm.npz"))
    lstm = Activations(str(act_dir / "sample.beats_lstm.npz"))
    fps = float(blstm.fps)
    print("blstm", blstm.shape, "lstm", lstm.shape, "fps", fps)

    out = Path("tests/golden/beats")
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(tag, values):
        values = np.asarray(values, dtype=float)
        target = out / ("%s.npy" % tag)
        np.save(target, values)
        shape = values.shape or (1,)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append("%s\t%s\t%s\n" % (tag, "x".join(str(d) for d in shape), digest))
        print("%-26s %-10s %s" % (tag, values.shape, digest[:12]))

    # ---- DBN -----------------------------------------------------------------
    processor = DBNBeatTrackingProcessor(fps=fps)
    save("dbn_default", processor(blstm))
    # beats on the first and last frame of a synthetic activation
    act = np.zeros(200) + 1e-4
    act[[0, 49, 99, 149, 199]] = 1
    save("dbn_edge_frames", processor(act))
    processor.correct = False
    save("dbn_no_correction", processor(blstm))
    processor.correct = True
    processor.threshold = 0.25
    save("dbn_threshold_025", processor(blstm))
    processor.threshold = 1
    save("dbn_threshold_1", processor(blstm))
    processor.threshold = 0
    save("dbn_empty_path", processor(np.array([0, 1, 0, 1, 0, 1])))

    # ---- forward (online) mode ----------------------------------------------
    online = DBNBeatTrackingProcessor(fps=100, online=True)
    save("dbn_forward_all", online.process_forward(lstm))
    online.reset()
    frames = [online.process_forward(np.atleast_1d(act), reset=False) for act in lstm]
    save("dbn_forward_frames", np.nonzero([f.size > 0 for f in frames])[0])
    frames = [online.process_forward(np.atleast_1d(act), reset=False) for act in lstm]
    save("dbn_forward_frames_noreset", np.nonzero([f.size > 0 for f in frames])[0])

    # ---- tracking / detection -----------------------------------------------
    save("tracker", BeatTrackingProcessor(fps=fps)(blstm))
    save("detector", BeatDetectionProcessor(fps=fps)(blstm))

    # ---- CRF -----------------------------------------------------------------
    crf = CRFBeatDetectionProcessor(fps=fps)
    save("crf", crf(blstm))
    intervals = crf.possible_intervals(blstm) if hasattr(crf, "possible_intervals") else None
    if intervals is None:
        # mirror what the processor computes internally
        tempi = crf.tempo_estimator.process(blstm)
        intervals = np.sort((fps * 60.0 / tempi[:, 0])[:crf.num_intervals])[::-1].astype(int)
    save("crf_intervals", intervals)

    # ---- the command line detections ----------------------------------------
    detections = root / "tests" / "data" / "detections"
    for tag, name in [("det_dbn", "sample.dbn_beat_tracker.txt"),
                      ("det_tracker", "sample.beat_tracker.txt"),
                      ("det_detector", "sample.beat_detector.txt"),
                      ("det_crf", "sample.crf_beat_detector.txt")]:
        values = []
        for line in (detections / name).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                values.append(float(line.split()[0]))
        save(tag, values)

    (out / "index.tsv").write_text("".join(rows), encoding="utf-8", newline="\n")
    print("wrote", out / "index.tsv", "with", len(rows), "cases")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
