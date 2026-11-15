"""Structural checks for the recurrent ONNX builders.

The numbers are verified against madmom through the engine (see the plan); these tests
only pin the shape of the graphs, so a refactor cannot silently drop the peepholes or fall
back to the ONNX LSTM operator.
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

onnx = pytest.importorskip("onnx")

import onnx_recurrent  # noqa: E402
import onnx_spec  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "third_party/madmom/madmom/models"
FAMILIES = ("rnn", "brnn", "bgru", "lstm", "blstm")


def first_model(family: str) -> Path | None:
    for path in sorted(MODELS.rglob("*.pkl")):
        try:
            model = onnx_spec.load_model(path)
        except Exception:
            continue
        if onnx_spec.model_family(model) == family:
            return path
    return None


def build(path: Path):
    model = onnx_spec.load_model(path)
    family = onnx_spec.model_family(model)
    spec = onnx_spec.spec_for(path, family=family)
    spec = onnx_spec.with_model_width(spec, model)
    return onnx_spec.build_model(model, spec), spec


@pytest.mark.skipif(not MODELS.exists(), reason="madmom submodule not checked out")
@pytest.mark.parametrize("family", FAMILIES)
def test_every_recurrent_family_builds_a_scan(family):
    path = first_model(family)
    if path is None:
        pytest.skip(f"no model of family {family}")
    model, spec = build(path)
    ops = [node.op_type for node in model.graph.node]
    assert "Scan" in ops, f"{family}: no Scan in {ops}"
    # The ONNX LSTM operator cannot carry madmom's peepholes.
    assert "LSTM" not in ops and "GRU" not in ops and "RNN" not in ops
    assert model.ir_version == onnx_spec.IR_VERSION


@pytest.mark.skipif(not MODELS.exists(), reason="madmom submodule not checked out")
@pytest.mark.parametrize("family", ("lstm", "blstm"))
def test_lstm_graphs_carry_peepholes(family):
    path = first_model(family)
    if path is None:
        pytest.skip(f"no model of family {family}")
    model, _ = build(path)
    scan = next(node for node in model.graph.node if node.op_type == "Scan")
    body = next(attribute.g for attribute in scan.attribute if attribute.name == "body")
    initializers = {tensor.name for tensor in body.initializer}
    # three gates with peepholes plus the cell state carrying two states
    peepholes = [name for name in initializers if name.endswith(".p")]
    assert len(peepholes) >= 3, sorted(initializers)
    assert len([i for i in scan.input if i != "input"]) == 2, "LSTM keeps (h, c) as state"


@pytest.mark.skipif(not MODELS.exists(), reason="madmom submodule not checked out")
def test_old_gru_models_start_from_their_stored_state():
    """Old pickles name the state `hid_init`; reading only `init` started them from zeros."""
    path = first_model("bgru")
    if path is None:
        pytest.skip("no bgru model")
    model = onnx_spec.load_model(path)
    layer = model.layers[0].fwd_layer
    stored = getattr(layer, "hid_init", None)
    assert stored is not None, "expected an old-style GRU model"
    _, tensor = onnx_recurrent._initial_state(int(stored.size), "h0", layer, "init", "hid_init")
    recovered = onnx.numpy_helper.to_array(tensor)
    assert np.allclose(recovered, np.asarray(stored, np.float32))
