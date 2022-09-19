import struct

import onnx
from qonnx.util.cleanup import cleanup
from qonnx.util.to_channels_last import to_channels_last
import onnxoptimizer

import numpy as np


def find_next_op(model, target):
    for node in model.graph.node:
        node_inputs = node.input
        if target in node_inputs:
            return node


def remove_node(model, mul_node, div_node):
    mul_node_output = mul_node.output[0]

    next_node_after_div = find_next_op(model, div_node.output[0])
    node_inputs = next_node_after_div.input
    for idx, node_in in enumerate(node_inputs):
        if node_in == div_node.output[0]:
            next_node_after_div.input[idx] = mul_node_output

    model.graph.node.remove(div_node)


def merge_nodes(model, mul_node, next_node):
    if next_node is None or next_node.op_type != "Relu":
        return

    div_node = find_next_op(model, next_node.output[0])
    if div_node.op_type != "Div":
        return

    print(f" - Merging {mul_node.name} with {div_node.name}")
    mul_param = None
    mul_param_name = mul_node.name + "_param0"
    div_param_name = div_node.name + "_param0"

    for param in model.graph.initializer:
        if param.name == mul_param_name:
            mul_param = param
            mul_data = struct.unpack(f"{int(len(param.raw_data)/4)}f", param.raw_data)
        if param.name == div_param_name:
            div_data = struct.unpack(f"{int(len(param.raw_data)/4)}f", param.raw_data)

    mul_data = np.array(mul_data)
    new_mul1_data = mul_data / div_data
    mul_param.raw_data = struct.pack(f"{len(new_mul1_data)}f", *list(new_mul1_data))

    remove_node(model, next_node, div_node)


def merge_quant_linear_scaling(model):
    for node in model.graph.node:
        if node.op_type == "Mul":
            next_node = find_next_op(model, node.output[0])
            merge_nodes(model, node, next_node)


def optimize_onnx_model(model_path):
    onnx_model = onnxoptimizer.optimize(
        onnx.load_model(model_path), passes=["extract_constant_to_initializer"]
    )
    cleanup(onnx_model, out_file=model_path)
    to_channels_last(model_path, out_file=model_path)

    onnx_model = onnx.load(model_path)
    merge_quant_linear_scaling(onnx_model)
    cleanup(onnx_model, out_file=model_path)


def gen_filename():
    from datetime import datetime

    now = datetime.now()  # current date and time
    date_time = now.strftime("%m%d%Y_%H%M%S")
    filename = f"hawq2qonnx_{date_time}.onnx"
    return filename
