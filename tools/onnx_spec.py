#!/usr/bin/env python3
"""Specification of the ONNX model repository derived from madmom's pickles.

This module is the executable form of section 2.1/2.2 of the porting plan:

* how a madmom pickle is read (the two NumPy 2 quirks are handled here and are
  covered by tests),
* what the model repository looks like (directory naming, tensor names,
  `config.pbtxt`),
* how madmom layers are expressed as ONNX nodes (gate order, gate polarity,
  axis layouts - the traps that silently break parity).

Nothing in here is madmom-specific glue: the pickles are read directly, so the
conversion does not need madmom to be installed.
"""
from __future__ import annotations

import contextlib
import importlib
import pickle
import re
import sys
import types
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# pickle loading
# ---------------------------------------------------------------------------
# All 92 model files are protocol-2 pickles written by Python 2 / NumPy 1.x.
# Two things must be respected (both verified against the real files):
#   1. Unpickler.find_class must not be replaced wholesale: doing so loses the
#      built-in py2->py3 module mapping (c_codecs -> codecs), which breaks 11
#      files with "Cannot interpret '<encode>' as a data type".
#   2. notes/2018/notes_cnn_[12].pkl reference numpy.lib.shape_base, which no
#      longer exists in NumPy 2; a module shim that proxies to numpy fixes it.

NUMPY_SHIM_MODULES = (
    "numpy.lib.shape_base",
    "numpy.lib.function_base",
    "numpy.core.numeric",
    "numpy.core.fromnumeric",
)


@contextlib.contextmanager
def numpy2_pickle_shims() -> Iterator[None]:
    """Make pickles written against NumPy 1.x load under NumPy 2."""
    created = {}
    for name in NUMPY_SHIM_MODULES:
        if name in sys.modules:
            continue
        try:  # the module may still exist under NumPy 2
            importlib.import_module(name)
            continue
        except Exception:
            pass
        module = types.ModuleType(name)
        module.__getattr__ = lambda attr, _np=np: getattr(_np, attr)
        if name == "numpy.lib.shape_base":
            module._array_from_pyobj = lambda arr, dtype, ndmin: np.array(
                arr, dtype=dtype, ndmin=int(ndmin)
            )
        sys.modules[name] = module
        created[name] = module
    try:
        yield
    finally:
        for name in created:
            sys.modules.pop(name, None)


class _Stub:
    """Stand-in for a madmom class so the object graph can be read."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._args = args
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __setstate__(self, state: Any) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self._state = state

    def __repr__(self) -> str:
        return "<%s>" % type(self).__name__


_STUB_CACHE: dict[str, type] = {}


def _stub_class(fullname: str) -> type:
    module, _, name = fullname.rpartition(".")
    if fullname not in _STUB_CACHE:
        _STUB_CACHE[fullname] = type(name, (_Stub,), {"__module__": module})
    return _STUB_CACHE[fullname]


class MadmomUnpickler(pickle.Unpickler):
    """Unpickler that stubs madmom classes and keeps stdlib compat mappings."""

    def find_class(self, module: str, name: str) -> Any:
        fullname = "%s.%s" % (module, name)
        if module.startswith("madmom"):
            if fullname == "madmom.ml.nn.activations.tanh":
                return np.tanh
            return _stub_class(fullname)
        return super().find_class(module, name)


def load_model(path: str | Path) -> Any:
    """Load a pickled madmom model (network, CRF, GMM, ...)."""
    with numpy2_pickle_shims():
        with open(path, "rb") as handle:
            with warnings.catch_warnings():
                # NumPy 2 warns that the alignment flag of the pickled dtype
                # descriptors is a Python int rather than a bool; the arrays load
                # correctly, so the warning is noise here.
                warnings.simplefilter("ignore", category=Warning)
                return MadmomUnpickler(handle, encoding="latin1").load()


def _walk(obj: Any, seen: set[int] | None = None, depth: int = 0) -> Iterator[Any]:
    seen = seen if seen is not None else set()
    if id(obj) in seen or depth > 40:
        return
    seen.add(id(obj))
    yield obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _walk(item, seen, depth + 1)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from _walk(key, seen, depth + 1)
            yield from _walk(value, seen, depth + 1)
    elif isinstance(obj, _Stub):
        for value in vars(obj).values():
            yield from _walk(value, seen, depth + 1)
        for value in getattr(obj, "_args", ()):
            yield from _walk(value, seen, depth + 1)


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------

@dataclass
class LayerInfo:
    """One layer of a network: class name plus its array attributes."""

    name: str
    module: str
    tensors: dict[str, tuple[int, ...]] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - debug helper
        return "%s(%s)" % (self.name, ", ".join("%s%s" % (k, v) for k, v in self.tensors.items()))


def layers_of(model: Any) -> list[LayerInfo]:
    """Layers of a network, in execution order, with their tensor shapes."""
    infos: list[LayerInfo] = []
    for candidate in _walk(model):
        name = type(candidate).__name__
        if not isinstance(candidate, _Stub) or name in ("NeuralNetwork", "NeuralNetworkEnsemble",
                                                       "SequentialProcessor", "ParallelProcessor"):
            continue
        info = LayerInfo(name=name, module=type(candidate).__module__)
        for key, value in sorted(vars(candidate).items()):
            if key.startswith("_"):
                continue
            if isinstance(value, np.ndarray):
                info.tensors[key] = tuple(int(d) for d in value.shape)
            elif isinstance(value, (int, float, bool, str)) or value is None:
                info.scalars[key] = value
        if info.tensors or info.scalars:
            infos.append(info)
    return infos


def model_family(model: Any) -> str:
    """Coarse family of a pickled model, used to pick a builder."""
    classes = {type(node).__name__ for node in _walk(model)}
    if "ConditionalRandomField" in classes:
        return "crf"
    if "GMM" in classes:
        return "gmm"
    if "TCNLayer" in classes or "TCNBlock" in classes:
        return "tcn"
    if "ConvolutionalLayer" in classes:
        return "cnn"
    if "BidirectionalLayer" in classes:
        if "GRULayer" in classes:
            return "bgru"
        if "LSTMLayer" in classes:
            return "blstm"
        return "brnn"
    if "LSTMLayer" in classes:
        return "lstm"
    if "GRULayer" in classes:
        return "gru"
    if "RecurrentLayer" in classes:
        return "rnn"
    if "FeedForwardLayer" in classes:
        return "feedforward"
    return "unknown"


# ---------------------------------------------------------------------------
# model repository layout (plan section 2.1)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Directory name, tensor names and input shape of one ONNX model."""

    name: str
    input_name: str = "input"
    output_names: tuple[str, ...] = ("output",)
    input_dims: tuple[int, ...] = (-1, 0)  # -1 = variable frame count
    output_dims: tuple[tuple[int, ...], ...] = ((-1, 1),)
    family: str = "unknown"

    def config_pbtxt(self, cpu_only: bool = True) -> str:
        """Render the Triton `config.pbtxt` for this model."""
        lines = [
            'name: "%s"' % self.name,
            'backend: "onnxruntime"',
            "max_batch_size: 0  # frames are batched by the caller, not by Triton",
        ]
        if cpu_only:
            lines.append("instance_group [{ kind: KIND_CPU }]")
        lines.append("input [")
        lines.append("  {")
        lines.append('    name: "%s"' % self.input_name)
        lines.append("    data_type: TYPE_FP32")
        lines.append("    dims: [ %s ]" % ", ".join(str(d) for d in self.input_dims))
        lines.append("  }")
        lines.append("]")
        lines.append("output [")
        for name, dims in zip(self.output_names, self.output_dims):
            lines.append("  {")
            lines.append('    name: "%s"' % name)
            lines.append("    data_type: TYPE_FP32")
            lines.append("    dims: [ %s ]" % ", ".join(str(d) for d in dims))
            lines.append("  }")
        lines.append("]")
        return "\n".join(lines) + "\n"


# Directory name -> (input feature count, number of outputs). Feature counts are
# the widths of the feature matrices the madmom preprocessors build (they are
# re-derived by tools/dump_preproc.py, which is the authority).
FEATURE_WIDTHS: dict[str, int] = {
    "beats_blstm": 162,   # 3 frame sizes x (6 bands * (1 spec + 1 diff)) = 3 * 27 * 2
    "beats_lstm": 162,
    "beats_tcn": 3,       # 3-D input: (frames, 12 bands, 1) after log
    "onsets_brnn": 162,
    "onsets_rnn": 162,
    "onsets_brnn_pp": 3,  # post-processing network on 1-3 bands
    "onsets_cnn": 3,
    "downbeats_blstm": 2,
    "chroma_dnn": 12,
    "notes_brnn": 88,
    "notes_cnn": 3,
    "key_cnn": 3,
    "chords_cnnfeat": 113,  # frequency bins the CNN sees; from CNNChordFeatureProcessor
}


def group_of(pickle_path: str | Path) -> str:
    """Group a pickle path belongs to, e.g. `beats_blstm_1` -> `beats_blstm`."""
    stem = Path(pickle_path).stem
    return re.sub(r"_?\d+$", "", stem)


def input_width_of(model: Any) -> int | None:
    """Input width of a network, taken from the weights of its first layer.

    FEATURE_WIDTHS is a hand-written table and had chroma_dnn at 12 while the pickled
    network expects 1575, so the graph and its config.pbtxt disagreed with the model. The
    weights are the authority; the table stays as the fallback for families whose first
    layer does not carry an (in, out) matrix.
    """
    for info in layers_of(model):
        shape = info.tensors.get("weights")
        if shape and len(shape) == 2:
            return int(shape[0])
    return None


def spec_for(pickle_path: str | Path, family: str | None = None) -> ModelSpec:
    """Model spec of the ONNX model built from `pickle_path`."""
    name = Path(pickle_path).stem
    group = group_of(pickle_path)
    width = FEATURE_WIDTHS.get(group, 0)
    dims: tuple[int, ...]
    if family in ("cnn", "tcn"):
        dims = (-1, width, 1)
    elif family == "blstm" and group.endswith("downbeats_blstm"):
        dims = (-1, width, 2)
    else:
        dims = (-1, width)
    outputs = ("output_beats", "output_tempo") if family == "tcn" else ("output",)
    output_dims = ((-1, 1), (-1, 1)) if family == "tcn" else ((-1, 1),)
    return ModelSpec(
        name=name,
        input_dims=dims,
        output_names=outputs,
        output_dims=output_dims,
        family=family or "unknown",
    )


MODEL_GROUPS = tuple(sorted(FEATURE_WIDTHS))


# ---------------------------------------------------------------------------
# ONNX graph builders
# ---------------------------------------------------------------------------
# Only the recurrent/feed-forward families are implemented here; the remaining
# ones (CNN, TCN, CRF, GMM) are reported as unsupported by convert_to_onnx.py so
# that the gap is visible instead of producing wrong models.
# NOTE: the graphs below are written against the ONNX helper API but have not
# been executed yet - no `onnx` package is available in the authoring
# environment. They must pass `--validate` (onnx.checker + numeric comparison
# against the reference activations) before they are trusted.

# Highest IR version the vendored ONNX Runtime (1.18.1, in the Triton backend) accepts.
IR_VERSION = 10
OPSET = 17


class UnsupportedLayer(Exception):
    """Raised for layers without an ONNX builder yet."""


def _require_onnx():
    try:
        import onnx  # noqa: F401
        from onnx import helper, numpy_helper  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "the onnx package is required for graph building: %s\n"
            "run e.g. nix-shell -p 'python3.withPackages(ps: [ps.numpy ps.onnx])'" % exc
        )
    import onnx
    from onnx import helper, numpy_helper
    return onnx, helper, numpy_helper


def _activation_node(helper, name: str, source: str, target: str) -> Any:
    kind = {"tanh": "Tanh", "sigmoid": "Sigmoid", "relu": "Relu"}.get(name)
    if kind is None:
        if name == "elu":  # madmom: x < 0 ? exp(x) - 1 : x
            return helper.make_node("Elu", [source], [target], alpha=1.0)
        if name in ("linear", "None", ""):
            return helper.make_node("Identity", [source], [target])
        raise UnsupportedLayer("activation %r" % name)
    return helper.make_node(kind, [source], [target])


def _activation_name(layer: Any) -> str:
    fn = getattr(layer, "activation_fn", None)
    if fn is None:
        return "linear"
    return getattr(fn, "__name__", str(fn))


def build_feedforward(layers: Iterable[Any], input_width: int, spec: ModelSpec) -> Any:
    """MatMul + Add (+ activation) per feed-forward layer."""
    onnx, helper, numpy_helper = _require_onnx()
    layers = list(layers)
    initializers, nodes = [], []
    current, width = spec.input_name, input_width
    for index, layer in enumerate(layers):
        weights = np.asarray(layer.weights, dtype=np.float32)
        bias = np.asarray(layer.bias, dtype=np.float32).reshape(-1)
        w_name, b_name = "layer%d.weights" % index, "layer%d.bias" % index
        initializers.append(numpy_helper.from_array(weights, w_name))
        initializers.append(numpy_helper.from_array(bias, b_name))
        matmul, added = "layer%d.matmul" % index, "layer%d.add" % index
        nodes.append(helper.make_node("MatMul", [current, w_name], [matmul]))
        nodes.append(helper.make_node("Add", [matmul, b_name], [added]))
        activation = _activation_name(layer)
        output = spec.output_names[0] if index == len(layers) - 1 else "layer%d.out" % index
        if activation in ("linear", "None", ""):
            nodes.append(helper.make_node("Identity", [added], [output]))
        else:
            nodes.append(_activation_node(helper, activation, added, output))
        current, width = output, weights.shape[1]
    return _finish_graph(helper, onnx, nodes, initializers, spec, width)


def _lstm_gate_pack(layer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack madmom's four LSTM gates into ONNX's [i, o, f, c] order.

    madmom keeps the gates as separate objects in the order input, forget, cell,
    output; ONNX concatenates them along axis 0 in the order i, o, f, c. Getting
    this wrong produces a model that still runs and gives wrong numbers.
    """
    order = ("input_gate", "output_gate", "forget_gate", "cell")
    weights = np.concatenate([np.asarray(getattr(layer, g).weights, dtype=np.float32)
                              for g in order], axis=1)
    recurrent = np.concatenate([np.asarray(getattr(layer, g).recurrent_weights, dtype=np.float32)
                                for g in order], axis=1)
    bias = np.concatenate([np.asarray(getattr(layer, g).bias, dtype=np.float32).reshape(-1)
                           for g in order])
    return weights.T, recurrent.T, bias


def build_lstm(layer: Any, input_width: int, spec: ModelSpec, *, name: str = "lstm",
               bidirectional: bool = False, reverse_layer: Any = None) -> Any:
    """Refuse to emit a plain ONNX LSTM node for a madmom `LSTMLayer`.

    madmom's LSTM gates carry **peephole** connections (the previous cell state,
    weighted per gate), and every LSTM/BLSTM model in madmom_models uses them:
    measured over the 92 model files, all 24 `blstm` (max |p| = 35.1) and all 8
    `lstm` (max |p| = 2.80) have non-zero peepholes.

    ONNX gives the LSTM operator an optional `P` (peephole) input, but
    onnxruntime does not implement it: a node with `P` fails to load with
    `INVALID_GRAPH: Type Error: Type 'tensor(float)' of input parameter (P) ...
    invalid`, while the identical node without `P` runs — both checked against
    onnxruntime 1.24.4. Emitting the node anyway would silently drop the
    peepholes and yield a model that runs and returns wrong numbers, which is the
    one failure mode this project refuses. The recurrence has to be spelled out
    with primitive ops (MatMul/Add/Sigmoid/Tanh/Mul, plus a Slice for the
    per-gate split) in plan task 2.3; `_lstm_gate_pack` and `_lstm_peepholes`
    below already produce the tensors that builder needs.

    The previous implementation of this function also mis-packed the node inputs
    (it passed `B` in the peephole slot and zeroed madmom's gate biases), so the
    numbers were wrong even before the peephole question.
    """
    peepholes = (_lstm_peepholes(reverse_layer, bidirectional=True, forward=layer)
                 if bidirectional else _lstm_peepholes(layer))
    magnitude = 0.0 if peepholes is None else float(np.abs(peepholes).max())
    if magnitude == 0.0:
        raise UnsupportedLayer(
            "%s: the ONNX LSTM node path was abandoned (see build_lstm's "
            "docstring); use the primitive-op recurrence of plan task 2.3" % name)
    raise UnsupportedLayer(
        "%s: peephole weights (max |p| = %.4g) cannot be expressed as an ONNX LSTM "
        "node - onnxruntime rejects the optional P input, and dropping them would "
        "silently change the numbers; build the recurrence from primitive ops in "
        "plan task 2.3" % (name, magnitude))


def _lstm_peepholes(layer: Any, *, bidirectional: bool = False, forward: Any = None) -> Any:
    """Peephole weights in ONNX order, or None if madmom has no peepholes."""
    order = ("input_gate", "output_gate", "forget_gate")
    parts = []
    for gate in order:
        value = getattr(getattr(layer, gate), "peephole_weights", None)
        if value is None:
            return None
        parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
    packed = np.concatenate(parts)
    if bidirectional:
        fwd_parts = []
        for gate in order:
            value = getattr(getattr(forward, gate), "peephole_weights", None)
            if value is None:
                return None
            fwd_parts.append(np.asarray(value, dtype=np.float32).reshape(-1))
        packed = np.concatenate([np.concatenate(fwd_parts), packed])
    return packed


def _finish_graph(helper, onnx, nodes, initializers, spec: ModelSpec, output_width: int) -> Any:
    graph = helper.make_graph(
        nodes,
        spec.name,
        [helper.make_tensor_value_info(spec.input_name, onnx.TensorProto.FLOAT,
                                       [d if d != 0 else None for d in spec.input_dims])],
        [helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, [-1, output_width])
         for name in spec.output_names],
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    # Pin the IR version: helper.make_model stamps whatever the installed onnx package
    # declares (13 with onnx 1.21), while the ONNX Runtime vendored into the Triton backend
    # is 1.18.1 and refuses anything above IR 10 with "Unsupported model IR version".
    model.ir_version = IR_VERSION
    model.doc_string = "converted from madmom pickle by tools/convert_to_onnx.py"
    return model


BUILDERS = {
    "feedforward": build_feedforward,
    "lstm": build_lstm,
}


def output_width_of(model: Any) -> int | None:
    """Output width of a feed-forward network, from the weights of its last layer."""
    width = None
    for info in layers_of(model):
        shape = info.tensors.get("weights")
        if shape and len(shape) == 2:
            width = int(shape[1])
    return width


def with_model_width(spec: ModelSpec, model: Any) -> ModelSpec:
    """`spec` with the input width taken from the weights of the model.

    Callers that write a `config.pbtxt` must use the returned spec: the width in the table
    and the width the graph is built for have to be the same number.
    """
    width = input_width_of(model)
    if width is None or len(spec.input_dims) < 2 or spec.input_dims[1] == width:
        return spec
    dims = list(spec.input_dims)
    dims[1] = width
    out = spec.output_dims
    if spec.family not in ("tcn",) and out == ((-1, 1),):
        # chroma_dnn answers with 12 chroma bins and notes_brnn with 88, not one value: the
        # default has to give way to the model whenever the last layer says something else.
        produced = output_width_of(model)
        if produced is not None:
            out = ((-1, produced),)
    return ModelSpec(**{**spec.__dict__, "input_dims": tuple(dims), "output_dims": out})


def build_model(model: Any, spec: ModelSpec) -> Any:
    """Build the ONNX model for a loaded madmom pickle."""
    family = model_family(model)
    if family != spec.family:
        spec = ModelSpec(**{**spec.__dict__, "family": family})
    spec = with_model_width(spec, model)
    if family in ("lstm", "blstm", "gru", "bgru", "rnn", "brnn"):
        # Primitive-op recurrence (tools/onnx_recurrent.py): the ONNX LSTM operator cannot
        # express madmom's peephole connections, onnxruntime does not implement its P input.
        import onnx_recurrent
        return onnx_recurrent.build(model, spec)
    if family == "cnn":
        # Convolutions with flipped kernels in NCHW (tools/onnx_conv.py); madmom convolves,
        # ONNX correlates, and the difference is invisible until the numbers are compared.
        import onnx_conv
        return onnx_conv.build(model, spec)
    if family in BUILDERS:
        layers = [node for node in getattr(model, "layers", []) if isinstance(node, _Stub)]
        return BUILDERS[family](layers, spec.input_dims[-1], spec)
    raise UnsupportedLayer("model family %r has no ONNX builder yet (plan phase 2/4)" % family)


def _build_recurrent_family(model: Any, family: str, spec: ModelSpec) -> Any:
    layers = list(getattr(model, "layers", []))
    if family == "lstm":
        return build_lstm(layers[0], spec.input_dims[-1], spec)
    if family == "blstm":
        wrapper = layers[0]
        return build_lstm(getattr(wrapper, "fwd_layer"), spec.input_dims[-1], spec,
                          bidirectional=True, reverse_layer=getattr(wrapper, "bwd_layer"))
    raise UnsupportedLayer(
        "recurrent family %r: implement the ONNX RNN/GRU packing in phase 2 (see plan section 2.1)"
        % family
    )
