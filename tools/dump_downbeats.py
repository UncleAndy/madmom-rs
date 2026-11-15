#!/usr/bin/env python3
"""Dump the reference joint beat/downbeat DBN output for the downbeat fixture.

Uses the *reference activations* (`tests/data/activations/sample.downbeats_blstm.npz`,
the output of madmom's own downbeat network) so the network is not needed: the dump
pins the DBN, which is what the port has to reproduce.

Writes into `tests/golden/downbeats/`:
  result.npy          -- (num_beats, 2) positions [seconds] and beat numbers
  path_<bpb>.npy      -- Viterbi state path for every modelled bar length
  logprob_<bpb>.npy   -- its log probability (scalar)
and `index.tsv` with `tag`, `shape`, `sha256`.

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.mido])' --run \\
      'python3 tools/dump_downbeats.py third_party/madmom'
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

BEATS_PER_BAR = (3, 4)


def main(argv):
    if len(argv) != 2:
        raise SystemExit(__doc__)
    madmom_root = Path(argv[1]).resolve()
    sys.path.insert(0, str(madmom_root))
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor

    fixture = madmom_root / "tests" / "data" / "activations" / "sample.downbeats_blstm.npz"
    with np.load(fixture) as data:
        activations = np.asarray(data["activations"], dtype=float)
        fps = float(np.asarray(data["fps"]))
    print("activations", activations.shape, "fps", fps)

    out = Path("tests/golden/downbeats")
    out.mkdir(parents=True, exist_ok=True)
    processor = DBNDownBeatTrackingProcessor(beats_per_bar=list(BEATS_PER_BAR), fps=fps)
    result = processor(activations)
    rows = []

    def save(tag, values):
        target = out / ("%s.npy" % tag)
        np.save(target, np.asarray(values))
        shape = np.asarray(values).shape or (1,)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        rows.append("%s\t%s\t%s\n" % (tag, "x".join(str(dim) for dim in shape), digest))
        print("%-16s %-10s %s" % (tag, np.asarray(values).shape, digest[:12]))

    save("result", result)
    # the processor thresholds before decoding, so dump that stage as well: the trimmed
    # activations and the paths decoded from them (the untrimmed paths are kept too,
    # because they are what a caller sees when threshold=0)
    from madmom.features.beats import threshold_activations

    trimmed, first = threshold_activations(activations, processor.threshold)
    print("threshold %.2f: rows %d..%d of %d kept" % (processor.threshold, first,
                                                      first + len(trimmed) - 1, len(activations)))
    save("trimmed", trimmed)
    for beats_per_bar, hmm in zip(BEATS_PER_BAR, processor.hmms):
        path, log_prob = hmm.viterbi(trimmed)
        save("path_trimmed_%d" % beats_per_bar, path)
        save("logprob_trimmed_%d" % beats_per_bar, np.array([log_prob]))
    for beats_per_bar, hmm in zip(BEATS_PER_BAR, processor.hmms):
        path, log_prob = hmm.viterbi(activations)
        save("path_%d" % beats_per_bar, path)
        save("logprob_%d" % beats_per_bar, np.array([log_prob]))
        print("   bar %d: %d states, log probability %.6f" %
              (beats_per_bar, hmm.transition_model.num_states, log_prob))
    (out / "index.tsv").write_text("".join(rows), encoding="utf-8", newline="\n")
    print("wrote", out / "index.tsv", "with", len(rows), "cases")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
