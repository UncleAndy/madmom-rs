"""ONNX graphs for madmom's recurrent layers.

The networks are built from primitive operations inside an ONNX `Scan`: madmom's LSTM
gates carry peephole connections, and the `P` input of the ONNX LSTM operator is not
implemented by onnxruntime, so the operator itself cannot express these models. Spell
the recurrence out instead, in exactly the order `madmom/ml/nn/layers.py` uses.

Every layer is one `Scan` whose body computes a single time step from the previous
state; bidirectional layers run the same body twice with `scan_input_directions=[1]`
and reverse the result back, which keeps the outputs aligned in time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from onnx import TensorProto, helper, numpy_helper

FLOAT = TensorProto.FLOAT

_ACTIVATIONS = {
    "tanh": "Tanh",
    "sigmoid": "Sigmoid",
    "relu": "Relu",
    "elu": "Elu",
}


def _activation(fn: Any) -> str | None:
    return _ACTIVATIONS.get(getattr(fn, "__name__", ""))


def _apply(nodes: list, fn: Any, source: str, target: str) -> None:
    """Append the activation `fn` (Identity when madmom uses none)."""
    kind = _activation(fn)
    if kind is None:
        nodes.append(helper.make_node("Identity", [source], [target]))
    elif kind == "Elu":
        nodes.append(helper.make_node(kind, [source], [target], alpha=1.0))
    else:
        nodes.append(helper.make_node(kind, [source], [target]))


def _init(name: str, array: Any) -> Any:
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def _matmul_add(nodes: list, x: str, w: str, h: str, r: str, b: str, out: str) -> None:
    """out = x @ w + h @ r + b, the shared shape of every gate pre-activation."""
    nodes.append(helper.make_node("MatMul", [x, w], [out + "_xw"]))
    nodes.append(helper.make_node("MatMul", [h, r], [out + "_hr"]))
    nodes.append(helper.make_node("Add", [out + "_xw", out + "_hr"], [out + "_sum"]))
    nodes.append(helper.make_node("Add", [out + "_sum", b], [out]))


def _scan(name: str, states: list[str], data: str, body_nodes: list, body_inits: list,
          state_shapes: list[tuple[int, ...]], input_width: int, out_width: int,
          reverse: bool = False, extra_inputs: list[tuple[str, tuple[int, ...]]] | None = None) -> Any:
    """One `Scan` node: `states` in, one scan input, states then the sequence out."""
    count = len(states)
    body_inputs = [
        helper.make_tensor_value_info(f"state{i}", FLOAT, shape)
        for i, shape in enumerate(state_shapes)
    ]
    for extra_name, extra_shape in extra_inputs or []:
        body_inputs.append(helper.make_tensor_value_info(extra_name, FLOAT, list(extra_shape)))
    body_inputs.append(helper.make_tensor_value_info("x_t", FLOAT, [input_width]))
    body_outputs = [
        helper.make_tensor_value_info(f"next{i}", FLOAT, shape)
        for i, shape in enumerate(state_shapes)
    ]
    body_outputs.append(helper.make_tensor_value_info("y_t", FLOAT, [out_width]))
    body = helper.make_graph(body_nodes, f"{name}_body", body_inputs, body_outputs,
                             initializer=body_inits)
    # Reverse by direction rather than by reversing the tensor: onnxruntime implements
    # scan_input_directions/scan_output_directions, while an explicit Range/Gather reversal
    # trips ORT's tensor slicer ("Invalid dim0_offset of 0") even though the graph is valid.
    # Both directions are reversed, so the output comes back aligned in time.
    direction = 1 if reverse else 0
    return helper.make_node(
        "Scan",
        list(states) + [data],
        [f"{name}_state{i}" for i in range(count)] + [f"{name}_seq"],
        body=body,
        num_scan_inputs=1,
        scan_input_axes=[0],
        scan_input_directions=[direction],
        scan_output_axes=[0],
        scan_output_directions=[direction],
    )


def _initial_state(hid: int, name: str, layer: Any, *attributes: str) -> tuple[str, Any]:
    """Initial state tensor, honouring the old attribute names in madmom's pickles.

    Old GRU models store `hid_init`; madmom renames it to `init` in `__setstate__`, which
    the stub classes used for conversion do not do. Reading only `init` silently started
    those networks from zeros and produced slightly wrong numbers everywhere.
    """
    for attribute in attributes:
        value = getattr(layer, attribute, None)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == hid:
            return name, _init(name, array)
    return name, _init(name, np.zeros(hid, dtype=np.float32))


def _rnn_layer(layer: Any, data: str, width: int, name: str, reverse: bool) -> tuple[list, list, str, int]:
    hid = int(np.asarray(layer.bias).size)
    w, r = f"{name}.W", f"{name}.R"
    b = f"{name}.b"
    inits = [_init(w, layer.weights), _init(r, layer.recurrent_weights), _init(b, layer.bias)]
    init_name, init = _initial_state(hid, f"{name}.h0", layer, "init", "hid_init")

    nodes: list = []
    _matmul_add(nodes, "x_t", w, "state0", r, b, f"{name}.pre")
    _apply(nodes, layer.activation_fn, f"{name}.pre", "y_t")
    nodes.append(helper.make_node("Identity", ["y_t"], ["next0"]))
    node = _scan(name, [init_name], data, nodes, inits, [(hid,)], width, hid, reverse=reverse)
    return [node], [init], f"{name}_seq", hid


def _gru_layer(layer: Any, data: str, width: int, name: str, reverse: bool) -> tuple[list, list, str, int]:
    hid = int(np.asarray(layer.cell.bias).size)
    inits: list = []
    states = []

    def gate(part: Any, tag: str) -> tuple[str, str, str]:
        w, r, b = f"{name}.{tag}.W", f"{name}.{tag}.R", f"{name}.{tag}.b"
        inits.extend([_init(w, part.weights), _init(r, part.recurrent_weights), _init(b, part.bias)])
        return w, r, b

    rw, rr, rb = gate(layer.reset_gate, "reset")
    uw, ur, ub = gate(layer.update_gate, "update")
    cw, cr, cb = gate(layer.cell, "cell")

    nodes: list = []
    _matmul_add(nodes, "x_t", rw, "state0", rr, rb, f"{name}.rg_pre")
    _apply(nodes, layer.reset_gate.activation_fn, f"{name}.rg_pre", f"{name}.rg")
    _matmul_add(nodes, "x_t", uw, "state0", ur, ub, f"{name}.ug_pre")
    _apply(nodes, layer.update_gate.activation_fn, f"{name}.ug_pre", f"{name}.ug")
    # GRUCell: x @ W + b + reset_gate * (prev @ R)
    nodes.append(helper.make_node("MatMul", ["x_t", cw], [f"{name}.cell_xw"]))
    nodes.append(helper.make_node("Add", [f"{name}.cell_xw", cb], [f"{name}.cell_b"]))
    nodes.append(helper.make_node("MatMul", ["state0", cr], [f"{name}.cell_hr"]))
    nodes.append(helper.make_node("Mul", [f"{name}.rg", f"{name}.cell_hr"], [f"{name}.cell_gated"]))
    nodes.append(helper.make_node("Add", [f"{name}.cell_b", f"{name}.cell_gated"], [f"{name}.cell_pre"]))
    _apply(nodes, layer.cell.activation_fn, f"{name}.cell_pre", f"{name}.cell")
    # out = update * cell + (1 - update) * prev
    nodes.append(helper.make_node("Mul", [f"{name}.ug", f"{name}.cell"], [f"{name}.out_a"]))
    nodes.append(helper.make_node("Sub", [f"{name}.one", f"{name}.ug"], [f"{name}.one_minus_ug"]))
    nodes.append(helper.make_node("Mul", [f"{name}.one_minus_ug", "state0"], [f"{name}.out_b"]))
    nodes.append(helper.make_node("Add", [f"{name}.out_a", f"{name}.out_b"], ["y_t"]))
    nodes.append(helper.make_node("Identity", ["y_t"], ["next0"]))
    inits.append(_init(f"{name}.one", np.ones(hid, dtype=np.float32)))

    init_name, init = _initial_state(hid, f"{name}.h0", layer, "init", "hid_init")
    node = _scan(name, [init_name], data, nodes, inits, [(hid,)], width, hid, reverse=reverse)
    return [node], [init], f"{name}_seq", hid


def _lstm_layer(layer: Any, data: str, width: int, name: str, reverse: bool) -> tuple[list, list, str, int]:
    hid = int(np.asarray(layer.cell.bias).size)
    inits: list = []

    def gate(part: Any, tag: str) -> dict[str, str]:
        names = {"w": f"{name}.{tag}.W", "r": f"{name}.{tag}.R", "b": f"{name}.{tag}.b",
                 "p": f"{name}.{tag}.p"}
        inits.extend([
            _init(names["w"], part.weights),
            _init(names["r"], part.recurrent_weights),
            _init(names["b"], part.bias),
            _init(names["p"], np.asarray(part.peephole_weights, np.float32).reshape(-1)),
        ])
        return names

    ig, fg = gate(layer.input_gate, "i"), gate(layer.forget_gate, "f")
    cg, og = gate(layer.cell, "c"), gate(layer.output_gate, "o")

    nodes: list = []
    # input gate: sigmoid(x W + h R + p_i * c_prev + b)
    _matmul_add(nodes, "x_t", ig["w"], "state0", ig["r"], ig["b"], f"{name}.i_pre")
    nodes.append(helper.make_node("Mul", [ig["p"], "state1"], [f"{name}.i_peep"]))
    nodes.append(helper.make_node("Add", [f"{name}.i_pre", f"{name}.i_peep"], [f"{name}.i_sum"]))
    _apply(nodes, layer.input_gate.activation_fn, f"{name}.i_sum", f"{name}.i")
    # forget gate: sigmoid(x W + h R + p_f * c_prev + b)
    _matmul_add(nodes, "x_t", fg["w"], "state0", fg["r"], fg["b"], f"{name}.f_pre")
    nodes.append(helper.make_node("Mul", [fg["p"], "state1"], [f"{name}.f_peep"]))
    nodes.append(helper.make_node("Add", [f"{name}.f_pre", f"{name}.f_peep"], [f"{name}.f_sum"]))
    _apply(nodes, layer.forget_gate.activation_fn, f"{name}.f_sum", f"{name}.f")
    # cell: tanh(x W + h R + b), no peepholes by construction
    _matmul_add(nodes, "x_t", cg["w"], "state0", cg["r"], cg["b"], f"{name}.c_pre")
    _apply(nodes, layer.cell.activation_fn, f"{name}.c_pre", f"{name}.cell")
    # c = cell * i + c_prev * f
    nodes.append(helper.make_node("Mul", [f"{name}.cell", f"{name}.i"], [f"{name}.c_new_a"]))
    nodes.append(helper.make_node("Mul", ["state1", f"{name}.f"], [f"{name}.c_new_b"]))
    nodes.append(helper.make_node("Add", [f"{name}.c_new_a", f"{name}.c_new_b"], ["next1"]))
    # output gate reads the *new* cell state
    _matmul_add(nodes, "x_t", og["w"], "state0", og["r"], og["b"], f"{name}.o_pre")
    nodes.append(helper.make_node("Mul", [og["p"], "next1"], [f"{name}.o_peep"]))
    nodes.append(helper.make_node("Add", [f"{name}.o_pre", f"{name}.o_peep"], [f"{name}.o_sum"]))
    _apply(nodes, layer.output_gate.activation_fn, f"{name}.o_sum", f"{name}.o")
    # h = tanh(c) * o
    nodes.append(helper.make_node("Tanh", ["next1"], [f"{name}.c_tanh"]))
    nodes.append(helper.make_node("Mul", [f"{name}.c_tanh", f"{name}.o"], ["y_t"]))
    nodes.append(helper.make_node("Identity", ["y_t"], ["next0"]))

    h0_name, h0 = _initial_state(hid, f"{name}.h0", layer, "init", "hid_init")
    c0_name, c0 = _initial_state(hid, f"{name}.c0", layer, "cell_init", "c_init")
    node = _scan(name, [h0_name, c0_name], data, nodes, inits, [(hid,), (hid,)], width, hid, reverse=reverse)
    return [node], [h0, c0], f"{name}_seq", hid


def _feed_forward(layer: Any, data: str, width: int, name: str) -> tuple[list, list, str, int]:
    w = np.asarray(layer.weights, np.float32)
    b = np.asarray(layer.bias, np.float32).reshape(-1)
    nodes = [
        helper.make_node("MatMul", [data, f"{name}.W"], [f"{name}.mm"]),
        helper.make_node("Add", [f"{name}.mm", f"{name}.b"], [f"{name}.pre"]),
    ]
    _apply(nodes, layer.activation_fn, f"{name}.pre", f"{name}.out")
    return nodes, [_init(f"{name}.W", w), _init(f"{name}.b", b)], f"{name}.out", int(w.shape[1])


def build(model: Any, spec: Any) -> Any:
    """Build the ONNX model for a recurrent madmom network."""
    layers = list(getattr(model, "layers", []))
    nodes: list = []
    inits: list = []
    current, width = spec.input_name, int(spec.input_dims[-1])
    for index, layer in enumerate(layers):
        name = f"layer{index}"
        kind = type(layer).__name__
        if kind == "BidirectionalLayer":
            inner = type(layer.fwd_layer).__name__
            builder = {"RecurrentLayer": _rnn_layer, "GRULayer": _gru_layer,
                       "LSTMLayer": _lstm_layer}[inner]
            fwd_nodes, fwd_inits, fwd_out, hid = builder(layer.fwd_layer, current, width,
                                                         name + ".fwd", False)
            # the backward layer scans the sequence in reverse and its output is reversed
            # back, so both directions line up in time before they are concatenated
            bwd_nodes, bwd_inits, bwd_out, _ = builder(layer.bwd_layer, current, width,
                                                       name + ".bwd", True)
            nodes.extend(fwd_nodes + bwd_nodes)
            inits.extend(fwd_inits + bwd_inits)
            nodes.append(helper.make_node("Concat", [f"{fwd_out}", f"{bwd_out}"],
                                          [f"{name}.out"], axis=1))
            current, width = f"{name}.out", hid * 2
        elif kind == "RecurrentLayer":
            produced, produced_inits, out, hid = _rnn_layer(layer, current, width, name, False)
            nodes.extend(produced)
            inits.extend(produced_inits)
            current, width = out, hid
        elif kind == "GRULayer":
            produced, produced_inits, out, hid = _gru_layer(layer, current, width, name, False)
            nodes.extend(produced)
            inits.extend(produced_inits)
            current, width = out, hid
        elif kind == "LSTMLayer":
            produced, produced_inits, out, hid = _lstm_layer(layer, current, width, name, False)
            nodes.extend(produced)
            inits.extend(produced_inits)
            current, width = out, hid
        elif kind == "FeedForwardLayer":
            produced, produced_inits, out, out_width = _feed_forward(layer, current, width, name)
            nodes.extend(produced)
            inits.extend(produced_inits)
            current, width = out, out_width
        else:
            raise onnx_spec_unsupported(kind)
    if current != spec.output_names[0]:
        nodes.append(helper.make_node("Identity", [current], [spec.output_names[0]]))
    return _finish(nodes, inits, spec, width)


def onnx_spec_unsupported(kind: str) -> Exception:
    from onnx_spec import UnsupportedLayer
    return UnsupportedLayer("recurrent network with a %r layer is not built yet" % kind)


def _finish(nodes: list, inits: list, spec: Any, width: int) -> Any:
    from onnx_spec import IR_VERSION, OPSET
    graph = helper.make_graph(
        nodes,
        spec.name,
        [helper.make_tensor_value_info(spec.input_name, FLOAT,
                                       [d if d != 0 else None for d in spec.input_dims])],
        [helper.make_tensor_value_info(spec.output_names[0], FLOAT, [-1, width])],
        initializer=inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    model.ir_version = IR_VERSION
    model.doc_string = "converted from madmom pickle by tools/convert_to_onnx.py"
    return model
