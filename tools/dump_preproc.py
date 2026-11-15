#!/usr/bin/env python3
"""Dump the *inputs and per-network outputs* of madmom's NN stages.

These dumps are the contract between the local preprocessing written in Rust
and the ONNX models served by Triton: the Rust side must reproduce the dumped
input byte-for-byte (within tolerance), and every converted model must
reproduce the dumped output of the corresponding network.

The preprocessor chain is taken from madmom itself (the first element of the
processor it composes), so this script cannot drift from the reference: it walks
the stages of the public processor until it reaches the network, dumps what the
network receives, then calls each network of the ensemble separately.

    nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.scipy ps.cython])' --run \
      'python3 tools/dump_preproc.py --audio third_party/madmom/tests/data/audio/sample.wav'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def groups():
    """Map a group name to a factory of the public madmom processor."""
    def make(kind):
        def factory():
            if kind == "beats_blstm":
                from madmom.features.beats import RNNBeatProcessor
                return RNNBeatProcessor()
            if kind == "beats_lstm":
                from madmom.features.beats import RNNBeatProcessor
                return RNNBeatProcessor(online=True)
            if kind == "beats_tcn":
                from madmom.features.beats import TCNBeatProcessor
                return TCNBeatProcessor()
            if kind == "onsets_brnn":
                from madmom.features.onsets import RNNOnsetProcessor
                return RNNOnsetProcessor()
            if kind == "onsets_cnn":
                from madmom.features.onsets import CNNOnsetProcessor
                return CNNOnsetProcessor()
            if kind == "downbeats_blstm":
                from madmom.features.downbeats import RNNDownBeatProcessor
                return RNNDownBeatProcessor()
            if kind == "chroma_dnn":
                from madmom.audio.chroma import DeepChromaProcessor
                return DeepChromaProcessor()
            raise SystemExit(f"group {kind} is not wired up yet (phase 2/4 of the plan)")
        return factory

    return {name: make(name) for name in
            ("beats_blstm", "beats_lstm", "beats_tcn", "onsets_brnn",
             "onsets_cnn", "downbeats_blstm", "chroma_dnn")}


def is_network(stage):
    """True for madmom network containers (ensemble or single network)."""
    from madmom.ml.nn import NeuralNetwork, NeuralNetworkEnsemble
    return isinstance(stage, (NeuralNetwork, NeuralNetworkEnsemble))


def networks_of(stage):
    """List the individual networks of a stage, in ensemble order."""
    from madmom.ml.nn import NeuralNetwork, NeuralNetworkEnsemble
    if isinstance(stage, NeuralNetwork):
        return [stage]
    if isinstance(stage, NeuralNetworkEnsemble):
        # SequentialProcessor((ParallelProcessor(networks), ensemble_fn))
        parallel = stage[0]
        return list(getattr(parallel, "processors", parallel))
    raise TypeError(f"not a network stage: {type(stage)}")


def dump_group(name, factory, audio, out_dir, max_networks):
    processor = factory()
    stages = list(getattr(processor, "processors", [processor]))
    data = audio
    for index, stage in enumerate(stages):
        if is_network(stage):
            features = np.ascontiguousarray(np.atleast_2d(data), dtype=np.float32)
            preproc_path = out_dir / "preproc" / f"{name}.npy"
            np.save(preproc_path, features)
            print(f"{name}: input to network {index} shape={features.shape} -> {preproc_path}")
            nets = networks_of(stage)
            for net_index, network in enumerate(nets):
                if max_networks is not None and net_index >= max_networks:
                    break
                output = np.asarray(network(features))
                net_name = name if len(nets) == 1 else f"{name}_{net_index + 1}"
                act_path = out_dir / "activations" / f"{net_name}.npy"
                np.save(act_path, output)
                print(f"{name}: network {net_index + 1} output shape={output.shape} -> {act_path}")
            return
        # madmom Processors are callable and simply forward to process()
        data = stage(data)
    raise SystemExit(f"{name}: no network stage found in the processor chain")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("tests/golden"))
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument("--max-networks", type=int, default=None,
                        help="dump only the first N networks of an ensemble")
    parser.add_argument("--list", action="store_true", help="list groups and exit")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    table = groups()
    selected = args.groups or list(table)
    unknown = [g for g in selected if g not in table]
    if unknown:
        raise SystemExit(f"unknown groups: {unknown}; known: {sorted(table)}")
    if args.list or args.dry_run:
        print("groups:", ", ".join(selected))
        print("input:", args.audio)
        print("output:", args.out)
        return 0

    (args.out / "preproc").mkdir(parents=True, exist_ok=True)
    (args.out / "activations").mkdir(parents=True, exist_ok=True)
    for name in selected:
        dump_group(name, table[name], args.audio, args.out, args.max_networks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
