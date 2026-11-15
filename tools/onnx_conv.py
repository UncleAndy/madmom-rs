"""ONNX graphs for madmom's convolutional layers.

madmom keeps data in `(frames, bins, channels)` and convolves with `scipy.ndimage.convolve`,
which is a true convolution — the kernel is flipped. ONNX `Conv` is a correlation, so every
kernel is flipped over its two spatial axes when it is emitted; without that the graph runs
and returns mirrored features.

The graph converts to NCHW once (`Transpose` + `Unsqueeze`), stays there, and converts back
to `(frames, bins * channels)` at the end, which is the shape the activation trait expects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto, helper, numpy_helper

FLOAT = TensorProto.FLOAT

_ACTIVATIONS = {"tanh": "Tanh", "sigmoid": "Sigmoid", "relu": "Relu", "elu": "Elu"}


def _init(name: str, array: Any) -> Any:
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _apply(nodes: list, fn: Any, source: str, target: str) -> None:
    kind = _ACTIVATIONS.get(getattr(fn, "__name__", ""))
    if kind is None:
        nodes.append(helper.make_node("Identity", [source], [target]))
    elif kind == "Elu":
        nodes.append(helper.make_node(kind, [source], [target], alpha=1.0))
    else:
        nodes.append(helper.make_node(kind, [source], [target]))


def _pair(value: Any, default: tuple[int, int] = (1, 1)) -> tuple[int, int]:
    """madmom stores these as None, an int, a tuple or a small numpy array."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        flat = [int(v) for v in value.reshape(-1)]
        return (flat[0], flat[1]) if len(flat) > 1 else (flat[0], flat[0])
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    value = int(value)
    return value, value


def conv_layer(layer: Any, data: str, name: str, inits: list, nodes: list) -> str:
    """One `Conv` (kernel flipped, madmom convolves rather than correlates)."""
    weights = np.asarray(layer.weights, dtype=np.float32)  # (channels, features, kt, kf)
    channels, features, kt, kf = weights.shape
    kernel = np.transpose(weights, (1, 0, 2, 3))[:, :, ::-1, ::-1]
    inits.append(_init(f"{name}.W", np.ascontiguousarray(kernel)))
    bias = np.asarray(layer.bias, dtype=np.float32).reshape(-1)
    if bias.size == 1:
        bias = np.repeat(bias, features)
    inits.append(_init(f"{name}.b", bias))
    stride = _pair(layer.stride)
    pads = [0, 0, 0, 0]
    if getattr(layer, "pad", "valid") == "same":
        pads = [kt // 2, kf // 2, (kt - 1) // 2, (kf - 1) // 2]
    elif getattr(layer, "pad", "valid") != "valid":
        raise ValueError(f"{name}: pad {layer.pad!r} is not supported")
    nodes.append(helper.make_node(
        "Conv", [data, f"{name}.W", f"{name}.b"], [f"{name}.conv"],
        kernel_shape=[kt, kf], strides=list(stride), pads=pads,
    ))
    _apply(nodes, getattr(layer, "activation_fn", None), f"{name}.conv", f"{name}.out")
    return f"{name}.out"


def batch_norm_layer(layer: Any, data: str, name: str, inits: list, nodes: list) -> str:
    """(x - mean) * (gamma * inv_std) + beta, per channel, as scale and shift."""
    # keep the shapes: a 2-D mean/inv_std means the layer normalises over (bins, channels)
    # rather than per channel, and flattening them first hid that (the check below could
    # never fire, and onsets_cnn got a (1, 240, 1, 1) scale instead of (1, 3, 1, 80))
    gamma = np.asarray(layer.gamma, np.float32)
    beta = np.asarray(layer.beta, np.float32)
    mean = np.asarray(layer.mean, np.float32)
    inv_std = np.asarray(layer.inv_std, np.float32)
    scale = gamma * inv_std
    shift = beta - mean * gamma * inv_std
    if mean.ndim == 2:
        # onsets_cnn normalises over (bins, channels), not per channel: the parameters are
        # (bins, channels) and map onto an NCHW tensor as (1, channels, 1, bins).
        scale = np.transpose(scale).reshape(1, scale.shape[1], 1, scale.shape[0])
        shift = np.transpose(shift).reshape(1, shift.shape[1], 1, shift.shape[0])
    else:
        scale = scale.reshape(1, -1, 1, 1)
        shift = shift.reshape(1, -1, 1, 1)
    inits.append(_init(f"{name}.scale", scale))
    inits.append(_init(f"{name}.shift", shift))
    nodes.append(helper.make_node("Mul", [data, f"{name}.scale"], [f"{name}.scaled"]))
    nodes.append(helper.make_node("Add", [f"{name}.scaled", f"{name}.shift"], [f"{name}.normed"]))
    _apply(nodes, getattr(layer, "activation_fn", None), f"{name}.normed", f"{name}.out")
    return f"{name}.out"


def max_pool_layer(layer: Any, data: str, name: str, nodes: list) -> str:
    """`maximum_filter` with zero padding, then the slice madmom takes: a valid pool."""
    if getattr(layer, "axis", None) is not None:
        raise ValueError(f"{name}: MaxPoolLayer with axis is not supported")
    size = _pair(layer.size)
    stride = _pair(getattr(layer, "stride", None))
    nodes.append(helper.make_node(
        "MaxPool", [data], [f"{name}.out"],
        kernel_shape=list(size), strides=list(stride), pads=[0, 0, 0, 0],
    ))
    return f"{name}.out"


def pad_layer(layer: Any, data: str, name: str, inits: list, nodes: list) -> str:
    """Constant padding in (frames, bins, channels) terms, applied to an NCHW tensor.

    madmom pads `width` on both sides of each listed axis; axis 0 is the frame axis and
    axis 1 the frequency axis, which are H and W here, and axis 2 is the channel axis.
    """
    axes = layer.axes if isinstance(layer.axes, (tuple, list, np.ndarray)) else [layer.axes]
    width = int(np.asarray(layer.width).reshape(-1)[0])
    axes = [int(np.asarray(axis).reshape(-1)[0]) for axis in axes]
    # NCHW index of each madmom axis
    nchw = {0: 2, 1: 3, 2: 1}
    pads = [0, 0, 0, 0, 0, 0, 0, 0]
    for axis in axes:
        position = nchw.get(axis)
        if position is None:
            raise ValueError(f"{name}: cannot pad axis {axis}")
        pads[position] = width
        pads[position + 4] = width
    value = np.asarray(layer.value, dtype=np.float32).reshape(())
    inits.append(_init(f"{name}.value", value))
    inits.append(numpy_helper.from_array(np.array(pads, dtype=np.int64), f"{name}.pads"))
    nodes.append(helper.make_node("Pad", [data, f"{name}.pads", f"{name}.value"], [f"{name}.out"],
                                  mode="constant"))
    return f"{name}.out"


# madmom axis in (frames, bins, channels) -> axis in NCHW
_AXIS_IN_NCHW = {0: 2, 1: 3, 2: 1}


def transpose_layer(layer: Any, data: str, name: str, nodes: list) -> str:
    """np.transpose over the madmom axes, translated to NCHW (batch axis stays first)."""
    axes = [int(np.asarray(axis).reshape(-1)[0])
            for axis in (layer.axes if isinstance(layer.axes, (tuple, list, np.ndarray)) else [layer.axes])]
    perm = [0] + [_AXIS_IN_NCHW[axis] for axis in axes]
    nodes.append(helper.make_node("Transpose", [data], [f"{name}.out"], perm=perm))
    return f"{name}.out"


def average_layer(layer: Any, data: str, name: str, nodes: list) -> str:
    """np.mean over one or more madmom axes; `keepdims` is preserved."""
    axes = layer.axis if isinstance(layer.axis, (tuple, list, np.ndarray)) else [layer.axis]
    axes = sorted({_AXIS_IN_NCHW[int(np.asarray(axis).reshape(-1)[0])] for axis in axes})
    keepdims = 1 if getattr(layer, "keepdims", False) else 0
    nodes.append(helper.make_node("ReduceMean", [data], [f"{name}.out"],
                                  axes=axes, keepdims=keepdims))
    return f"{name}.out"


def stride_layer(layer: Any, data: str, name: str, nodes: list, inits: list) -> str:
    """madmom's segment_axis along the frame axis with end='cut', then a flatten.

    In NCHW that is (C, n, B) -> (C, n//block, block, B) -> (C, n//block, block*B).
    """
    block = int(np.asarray(layer.block_size).reshape(-1)[0])
    nodes.append(helper.make_node("Shape", [data], [f"{name}.shape"]))
    nodes.append(helper.make_node("Gather", [f"{name}.shape", "_two_i64"], [f"{name}.frames"], axis=0))
    nodes.append(helper.make_node("Div", [f"{name}.frames", _const(inits, f"{name}.block", np.int64(block))],
                                  [f"{name}.windows"]))
    nodes.append(helper.make_node("Gather", [f"{name}.shape", "_zero_i64"], [f"{name}.channels_s"], axis=0))
    # Concat needs 1-D inputs; a Gather with a scalar index yields a 0-D tensor
    nodes.append(helper.make_node("Unsqueeze", [f"{name}.channels_s", "_zero_axes"], [f"{name}.channels"]))
    nodes.append(helper.make_node("Gather", [f"{name}.shape", "_three_i64"], [f"{name}.bins"], axis=0))
    nodes.append(helper.make_node("Unsqueeze", [f"{name}.windows", "_zero_axes"], [f"{name}.windows1"]))
    nodes.append(helper.make_node("Unsqueeze", [f"{name}.block", "_zero_axes"], [f"{name}.block1"]))
    nodes.append(helper.make_node("Unsqueeze", [f"{name}.bins", "_zero_axes"], [f"{name}.bins1"]))
    nodes.append(helper.make_node("Concat", [f"{name}.channels", f"{name}.windows1",
                                             f"{name}.block1", f"{name}.bins1"],
                                  [f"{name}.split_shape"], axis=0))
    nodes.append(helper.make_node("Reshape", [data, f"{name}.split_shape"], [f"{name}.split"]))
    nodes.append(helper.make_node("Mul", [f"{name}.block", f"{name}.bins"], [f"{name}.flat"]))
    nodes.append(helper.make_node("Unsqueeze", [f"{name}.flat", "_zero_axes"], [f"{name}.flat1"]))
    nodes.append(helper.make_node("Concat", [f"{name}.channels", f"{name}.windows1", f"{name}.flat1"],
                                  [f"{name}.out_shape"], axis=0))
    nodes.append(helper.make_node("Reshape", [f"{name}.split", f"{name}.out_shape"], [f"{name}.out"]))
    return f"{name}.out"


def _const(inits: list, name: str, value: Any) -> str:
    inits.append(numpy_helper.from_array(np.asarray(value), name))
    return name


def to_flat(nodes: list, data: str) -> str:
    """NCHW -> (frames, bins * channels), the shape the tail of these networks works in."""
    nodes.append(helper.make_node("Squeeze", [data, "_zero_axes"], ["flat_chw"]))
    nodes.append(helper.make_node("Transpose", ["flat_chw"], ["flat_fbc"], perm=[1, 2, 0]))
    nodes.append(helper.make_node("Shape", ["flat_fbc"], ["flat_shape"]))
    nodes.append(helper.make_node("Slice", ["flat_shape", "_one_i64", "_max_i64"], ["flat_dims"]))
    nodes.append(helper.make_node("ReduceProd", ["flat_dims"], ["flat_prod"], keepdims=1))
    nodes.append(helper.make_node("Concat", ["_neg_one_i64", "flat_prod"], ["flat_target"], axis=0))
    nodes.append(helper.make_node("Reshape", ["flat_fbc", "flat_target"], ["flat"]))
    return "flat"


def _shape_dim(dim: Any) -> int:
    if isinstance(dim, (int, np.integer)):
        return int(dim)
    value = np.asarray(dim).reshape(-1)
    return int(value[0]) if value.size else -1


def reshape_layer(layer: Any, data: str, name: str, inits: list, nodes: list) -> str:
    shape = [_shape_dim(dim) for dim in layer.newshape]
    inits.append(numpy_helper.from_array(np.array(shape, dtype=np.int64), f"{name}.shape"))
    nodes.append(helper.make_node("Reshape", [data, f"{name}.shape"], [f"{name}.out"]))
    return f"{name}.out"


def feed_forward_layer(layer: Any, data: str, name: str, inits: list, nodes: list) -> str:
    weights = np.asarray(layer.weights, np.float32)
    bias = np.asarray(layer.bias, np.float32).reshape(-1)
    inits.append(_init(f"{name}.W", weights))
    inits.append(_init(f"{name}.b", bias))
    nodes.append(helper.make_node("MatMul", [data, f"{name}.W"], [f"{name}.mm"]))
    nodes.append(helper.make_node("Add", [f"{name}.mm", f"{name}.b"], [f"{name}.pre"]))
    _apply(nodes, getattr(layer, "activation_fn", None), f"{name}.pre", f"{name}.out")
    return f"{name}.out"


def build(model: Any, spec: Any) -> Any:
    """Build the ONNX model for a convolutional madmom network."""
    kind = type(model).__name__
    if kind != "NeuralNetwork":
        # Several "cnn" pickles are not networks at all: notes_cnn is a SequentialProcessor
        # holding convolutions plus a ParallelProcessor. Reading `.layers` off such a wrapper
        # silently builds a different graph, so refuse instead of guessing.
        raise unsupported(kind)
    layers = list(getattr(model, "layers", []))
    nodes: list = []
    inits: list = []
    inits.extend([
        numpy_helper.from_array(np.array(0, dtype=np.int64), "_zero_i64"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "_one_i64"),
        numpy_helper.from_array(np.array([2**62], dtype=np.int64), "_max_i64"),
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "_neg_one_i64"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "_zero_axes"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "_zero_i64_scalar"),
        numpy_helper.from_array(np.array(3, dtype=np.int64), "_three_i64"),
        numpy_helper.from_array(np.array(2, dtype=np.int64), "_two_i64"),
    ])
    _pair_dim = int(np.asarray(spec.input_dims[-1]))
    # (frames, bins, channels) -> (1, channels, frames, bins)
    nodes.append(helper.make_node("Transpose", [spec.input_name], ["nchw_t"], perm=[2, 0, 1]))
    nodes.append(helper.make_node("Unsqueeze", ["nchw_t", "_zero_axes"], ["nchw"]))
    current = "nchw"
    flattened = False
    for index, layer in enumerate(layers):
        name = f"layer{index}"
        kind = type(layer).__name__
        if not flattened and kind in ("ReshapeLayer", "FeedForwardLayer"):
            current = to_flat(nodes, current)
            flattened = True
        if kind == "ConvolutionalLayer":
            current = conv_layer(layer, current, name, inits, nodes)
        elif kind == "BatchNormLayer":
            current = batch_norm_layer(layer, current, name, inits, nodes)
        elif kind == "MaxPoolLayer":
            current = max_pool_layer(layer, current, name, nodes)
        elif kind == "PadLayer":
            current = pad_layer(layer, current, name, inits, nodes)
        elif kind == "TransposeLayer":
            current = transpose_layer(layer, current, name, nodes)
        elif kind == "AverageLayer":
            current = average_layer(layer, current, name, nodes)
        elif kind == "StrideLayer":
            current = stride_layer(layer, current, name, nodes, inits)
        elif kind == "ReshapeLayer":
            current = reshape_layer(layer, current, name, inits, nodes)
        elif kind == "FeedForwardLayer":
            current = feed_forward_layer(layer, current, name, inits, nodes)
        else:
            raise unsupported(kind)
    last = type(layers[-1]).__name__ if layers else ""
    collapses_frames = (
        last == "AverageLayer"
        and not getattr(layers[-1], "keepdims", False)
        and any(int(np.asarray(axis).reshape(-1)[0]) == 0
                for axis in (layers[-1].axis if isinstance(layers[-1].axis, (tuple, list, np.ndarray))
                             else [layers[-1].axis]))
    )
    if flattened or collapses_frames:
        # a network that averages over the frame axis answers once per input, not per frame
        features = int(np.asarray(layers[-1].beta).size) if last == "BatchNormLayer" else 0
        if collapses_frames and features:
            inits.append(numpy_helper.from_array(np.array([1, features], dtype=np.int64),
                                                 "_model_shape"))
            nodes.append(helper.make_node("Reshape", [current, "_model_shape"],
                                          [spec.output_names[0]]))
        else:
            nodes.append(helper.make_node("Identity", [current], [spec.output_names[0]]))
        return finish(nodes, inits, spec)
    # back to (frames, bins * channels)
    nodes.append(helper.make_node("Squeeze", [current, "_zero_axes"], ["chw"]))
    nodes.append(helper.make_node("Transpose", ["chw"], ["fbch"], perm=[1, 2, 0]))
    nodes.append(helper.make_node("Shape", ["fbch"], ["shape"]))
    nodes.append(helper.make_node("Slice", ["shape", "_one_i64", "_max_i64"], ["dims"]))
    nodes.append(helper.make_node("ReduceProd", ["dims"], ["prod"], keepdims=1))
    nodes.append(helper.make_node("Concat", ["_neg_one_i64", "prod"], ["target"], axis=0))
    nodes.append(helper.make_node("Reshape", ["fbch", "target"], [spec.output_names[0]]))
    return finish(nodes, inits, spec)


def unsupported(kind: str) -> Exception:
    from onnx_spec import UnsupportedLayer
    return UnsupportedLayer("convolutional network with a %r layer is not built yet" % kind)


def finish(nodes: list, inits: list, spec: Any) -> Any:
    from onnx_spec import IR_VERSION, OPSET
    graph = helper.make_graph(
        nodes,
        spec.name,
        [helper.make_tensor_value_info(spec.input_name, FLOAT,
                                       [d for d in spec.input_dims if d != 0])],
        [helper.make_tensor_value_info(spec.output_names[0], FLOAT, [None, None])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model.doc_string = "converted from madmom pickle by tools/convert_to_onnx.py"
    return model
