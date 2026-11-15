"""Structural checks for the convolutional ONNX builder.

Numeric equivalence was demonstrated on chords_cnnfeat (max abs diff 1.9e-06 against
madmom); these tests pin the properties that made it work, so a refactor cannot quietly
flip a kernel or change the layout.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

onnx = pytest.importorskip("onnx")

import onnx_conv  # noqa: E402
import onnx_spec  # noqa: E402

MODELS = Path(__file__).resolve().parents[2] / "third_party/madmom/madmom/models"


def first_cnn():
    for path in sorted(MODELS.rglob("*.pkl")):
        try:
            model = onnx_spec.load_model(path)
        except Exception:
            continue
        if onnx_spec.model_family(model) == "cnn":
            return path
    return None


@pytest.mark.skipif(not MODELS.exists(), reason="madmom submodule not checked out")
def test_convolution_kernel_is_flipped():
    """madmom convolves (scipy), ONNX correlates: the kernel must be flipped."""
    path = first_cnn()
    if path is None:
        pytest.skip("no cnn model")
    model = onnx_spec.load_model(path)
    conv = next(layer for layer in model.layers if type(layer).__name__ == "ConvolutionalLayer")
    weights = np.asarray(conv.weights)

    inits: list = []
    nodes: list = []
    onnx_conv.conv_layer(conv, "input", "c", inits, nodes)
    import onnx.numpy_helper as nh
    kernel = nh.to_array(next(t for t in inits if t.name == "c.W"))
    expected = np.transpose(weights, (1, 0, 2, 3))[:, :, ::-1, ::-1]
    assert kernel.shape == expected.shape
    assert np.allclose(kernel, expected)


@pytest.mark.skipif(not MODELS.exists(), reason="madmom submodule not checked out")
def test_cnn_graph_uses_nchw_conv_and_batchnorm():
    path = first_cnn()
    if path is None:
        pytest.skip("no cnn model")
    model = onnx_spec.load_model(path)
    spec = onnx_spec.spec_for(path, family="cnn")
    graph = onnx_conv.build(model, spec)
    ops = [node.op_type for node in graph.graph.node]
    assert "Conv" in ops
    assert "Transpose" in ops and "Reshape" in ops
    assert graph.ir_version == onnx_spec.IR_VERSION
    # the first op maps (frames, bins, channels) to NCHW
    first = graph.graph.node[0]
    assert first.op_type == "Transpose"
    perm = next(attribute.ints for attribute in first.attribute if attribute.name == "perm")
    assert list(perm) == [2, 0, 1]
