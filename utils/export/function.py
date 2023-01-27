import torch
import torch.autograd as autograd
from torch.onnx import register_custom_op_symbolic

domain_info = {"name": "hawq2qonnx", "version": 1}


class QuantFunc(autograd.Function):
    name = "Quant"

    @staticmethod
    def forward(
        ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode
    ):
        n = 2 ** (bit_width.numpy() - 1) - 1
        x_int = torch.clamp(torch.round((x / scale) + zero_point), -n - 1, n)
        output = x_int - zero_point
        output = output * scale
        return output

    @staticmethod
    def symbolic(
        g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode
    ):
        return g.op(
            f'{domain_info["name"]}::Quant',
            x,
            scale,
            zero_point,
            bit_width,
            signed_i=int(signed),
            narrow_i=int(narrow_range),
            rounding_mode_s=rounding_mode,
        )


class BinaryQuantFunc(autograd.Function):
    name = "BipolarQuant"

    @staticmethod
    def forward(
        ctx, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode
    ):
        return x

    @staticmethod
    def symbolic(
        g, x, scale, zero_point, bit_width, signed, narrow_range, rounding_mode
    ):
        return g.op(f'{domain_info["name"]}::BipolarQuant', x, scale)


class TruncFunc(autograd.Function):
    name = "Trunc"

    @staticmethod
    def forward(
        ctx, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode
    ):
        return torch.trunc(x)

    @staticmethod
    def symbolic(
        g, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode
    ):
        return g.op(
            f'{domain_info["name"]}::Trunc',
            x,
            scale,
            zero_point,
            input_bit_width,
            output_bit_width,
            rounding_mode_s=rounding_mode,
        )


class ConvFunc(autograd.Function):
    name = "Conv"

    @staticmethod
    def forward(
        ctx, x, quant_input, bias, layer, dilations, group, kernel_shape, pads, strides
    ):
        return layer.conv(x)

    @staticmethod
    def symbolic(
        g, x, quant_input, bias, layer, dilations, group, kernel_shape, pads, strides
    ):
        if(bias is None):
            return g.op(
                "Conv",
                x,
                quant_input,
                dilations_i=dilations,
                group_i=group,
                kernel_shape_i=kernel_shape,
                pads_i=pads,
                strides_i=strides,
            )
        else:
            return g.op(
                "Conv",
                x,
                quant_input,
                bias,
                dilations_i=dilations,
                group_i=group,
                kernel_shape_i=kernel_shape,
                pads_i=pads,
                strides_i=strides,
            )


class RoundFunc(autograd.Function):
    name = "Round"

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def symbolic(g, x):
        return g.op(f'{domain_info["name"]}::Round', x)


def get_quant_func(bit_width):
    if bit_width == 1:
        return BinaryQuantFunc
    return QuantFunc


onnx_funcs = [BinaryQuantFunc, QuantFunc, TruncFunc, RoundFunc, ConvFunc]


def register_custom_ops():
    for func in onnx_funcs:
        register_custom_op_symbolic(
            f'{domain_info["name"]}::{func.name}', func.symbolic, domain_info["version"]
        )
