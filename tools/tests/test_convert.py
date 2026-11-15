"""Tests for the ONNX conversion tooling.

The inventory and specification tests run with NumPy alone (they read the real
model pickles from the madmom submodule). Graph building needs the `onnx`
package, which is not available everywhere; those tests are skipped through
`pytest.importorskip` rather than being silently disabled.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import onnx_spec  # noqa: E402

MODELS = TOOLS.parents[0] / "third_party" / "madmom" / "madmom" / "models"


def pickles():
    assert MODELS.is_dir(), "run: git submodule update --init --recursive"
    return sorted(MODELS.rglob("*.pkl"))


def test_all_models_load_and_are_classified():
    found = pickles()
    assert len(found) == 92, "expected 92 model files, found %d" % len(found)
    families = {}
    for path in found:
        model = onnx_spec.load_model(path)
        family = onnx_spec.model_family(model)
        assert family != "unknown", "%s has no known family" % path.name
        families[family] = families.get(family, 0) + 1
    # the families needed by the flagship pipelines must be present. The
    # observed distribution over the 92 files is: blstm=24 bgru=12 brnn=17
    # cnn=10 lstm=8 rnn=8 tcn=8 crf=2 gmm=2 feedforward=1.
    expected = {"lstm": 8, "blstm": 24, "bgru": 12, "rnn": 8, "brnn": 17,
                "feedforward": 1, "cnn": 10, "tcn": 8, "crf": 2, "gmm": 2}
    assert families == expected, "family histogram changed: %r" % families


def test_beats_lstm_layer_inventory():
    model = onnx_spec.load_model(MODELS / "beats" / "2016" / "beats_lstm_1.pkl")
    layers = onnx_spec.layers_of(model)
    kinds = [layer.name for layer in layers]
    # beats_lstm_* has three stacked LSTM layers of 25 hidden units each
    assert kinds.count("LSTMLayer") == 3
    lstm = next(layer for layer in layers if layer.name == "LSTMLayer")
    assert lstm.tensors["init"] == (25,)
    assert lstm.tensors["cell_init"] == (25,)
    # the weights live on the gates, not on the layer wrapper
    gate = next(layer for layer in layers if layer.name == "Gate")
    assert gate.tensors["weights"] == (162, 25)
    assert gate.tensors["recurrent_weights"] == (25, 25)
    assert gate.tensors["bias"] == (25,)
    assert gate.tensors["peephole_weights"] == (25,)
    cell = next(layer for layer in layers if layer.name == "Cell")
    assert cell.tensors["weights"] == (162, 25)
    assert "peephole_weights" not in cell.tensors  # cells have no peepholes


def test_model_specs_follow_the_repository_layout():
    lstm = onnx_spec.spec_for("beats_lstm_1.pkl", family="lstm")
    assert lstm.name == "beats_lstm_1"
    assert lstm.input_name == "input"
    assert lstm.output_names == ("output",)
    assert lstm.input_dims[0] == -1

    tcn = onnx_spec.spec_for("beats_tcn_3.pkl", family="tcn")
    assert tcn.output_names == ("output_beats", "output_tempo")
    assert len(tcn.input_dims) == 3, "cnn/tcn models take (frames, freq, channels)"

    grouped = onnx_spec.spec_for("onsets_brnn_pp_7.pkl", family="blstm")
    assert grouped.name == "onsets_brnn_pp_7"


def test_config_pbtxt_is_cpu_only_and_functional():
    config = onnx_spec.spec_for("beats_blstm_1.pkl", family="blstm").config_pbtxt()
    assert 'name: "beats_blstm_1"' in config
    assert 'backend: "onnxruntime"' in config
    assert "max_batch_size: 0" in config
    assert "KIND_CPU" in config
    assert 'data_type: TYPE_FP32' in config
    assert config.count("input [") == 1 and config.count("output [") == 1


def test_lstm_gate_order_is_i_o_f_c():
    """The gate packing is the trap that silently breaks parity."""
    model = onnx_spec.load_model(MODELS / "beats" / "2016" / "beats_lstm_1.pkl")
    lstm = next(layer for layer in onnx_spec.layers_of(model) if layer.name == "LSTMLayer")
    assert isinstance(lstm, onnx_spec.LayerInfo)
    raw = onnx_spec.load_model(MODELS / "beats" / "2016" / "beats_lstm_1.pkl")
    packed_w, packed_r, packed_b = onnx_spec._lstm_gate_pack(raw.layers[0])
    hidden = 25
    assert packed_w.shape == (4 * hidden, 162)
    assert packed_r.shape == (4 * hidden, hidden)
    assert packed_b.shape == (4 * hidden,)
    # input gate block first, output gate second (ONNX order), not forget second
    forget = np.asarray(raw.layers[0].forget_gate.weights)
    assert np.allclose(packed_w[hidden:2 * hidden].T, np.asarray(raw.layers[0].output_gate.weights))
    assert np.allclose(packed_w[2 * hidden:3 * hidden].T, forget)


def test_graph_building_needs_onnx_and_validates():
    onnx = pytest.importorskip("onnx", reason="onnx package is needed to build graphs")
    model = onnx_spec.load_model(MODELS / "chroma" / "2016" / "chroma_dnn.pkl")
    spec = onnx_spec.spec_for("chroma_dnn.pkl", family="feedforward")
    graph = onnx_spec.build_model(model, spec)
    assert graph.graph.name == "chroma_dnn"
    onnx.checker.check_model(graph)
