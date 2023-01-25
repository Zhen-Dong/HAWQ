import torch
import torch.nn as nn

from ..quantization_utils.quant_utils import (
    SymmetricQuantFunction,
    symmetric_linear_quantization_params,
)

from .function import get_quant_func
from .function import TruncFunc, RoundFunc, ConvFunc

from ..quantization_utils.quant_modules import (
    QuantAct,
    QuantDropout,
    QuantLinear,
    QuantBnConv2d,
)
from ..quantization_utils.quant_modules import (
    QuantMaxPool2d,
    QuantAveragePool2d,
    QuantConv2d,
)

SUPPORTED_LAYERS = (
    QuantAct,
    QuantLinear,
    QuantConv2d,
    QuantMaxPool2d,
    QuantAveragePool2d,
    QuantDropout,
    QuantBnConv2d,
)

model_info = dict()
model_info["dense_out"] = dict()
model_info["transformed"] = dict()

# ------------------------------------------------------------
class ExportQonnxQuantAct(nn.Module):
    def __init__(self, hawq_layer) -> None:
        super().__init__()
        self.hawq_layer = hawq_layer
        self.export_mode = False

        if self.hawq_layer.full_precision_flag:
            self.bit_width = 32
        else:
            self.bit_width = self.hawq_layer.activation_bit

        self.scale = (
            self.hawq_layer.act_scaling_factor.clone().detach().requires_grad_(False)
        )

    def __repr__(self):
        repr = (
            f"{self.__class__.__name__}(scale={self.scale.detach().item()}, bitwidth={self.hawq_layer.activation_bit},"
            + f" full_precision_flag={self.hawq_layer.full_precision_flag}, quant_mode={self.hawq_layer.quant_mode})"
        )
        return repr

    def forward(
        self,
        x,
        pre_act_scaling_factor=None,
        pre_weight_scaling_factor=None,
        identity=None,
        identity_scaling_factor=None,
        identity_weight_scaling_factor=None,
    ):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]
        if self.export_mode and pre_act_scaling_factor is None:
            pre_act_scaling_factor = torch.tensor([1.0], dtype=torch.float32)

        if self.export_mode:
            return (x, self.scale)
        else:
            x, act_scaling_factor = self.hawq_layer(
                x,
                pre_act_scaling_factor,
                pre_weight_scaling_factor,
                identity,
                identity_scaling_factor,
                identity_weight_scaling_factor,
            )
            return (x, act_scaling_factor)


# ------------------------------------------------------------
class ExportQonnxQuantLinear(nn.Module):
    def __init__(self, hawq_layer) -> None:
        super().__init__()
        self.hawq_layer = hawq_layer
        self.export_mode = False

        self.has_bias = hasattr(self.hawq_layer, "bias")
        in_features, out_features = (
            self.hawq_layer.weight.shape[1],
            self.hawq_layer.weight.shape[0],
        )
        self.fc = torch.nn.Linear(in_features, out_features, self.has_bias)
        self.fc.weight.data = torch.transpose(self.hawq_layer.weight_integer, 0, 1)
        if self.has_bias:
            self.fc.bias.data = self.hawq_layer.bias_integer

        self.scale = self.hawq_layer.fc_scaling_factor.clone().requires_grad_(False)
        self.weight_node_inputs = (
            torch.tensor(1, dtype=torch.float32),  # scale
            torch.tensor(0, dtype=torch.float32),  # zero point
            torch.tensor(self.hawq_layer.weight_bit, dtype=torch.float32),  # bit width
        )

        if self.has_bias:
            self.bias_node_inputs = (
                torch.tensor(1, dtype=torch.float32),  # scale
                torch.tensor(0, dtype=torch.float32),  # zero point
                torch.tensor(
                    self.hawq_layer.bias_bit, dtype=torch.float32
                ),  # bit width
            )

        self.node_attributes = (
            int(1 if self.hawq_layer.quant_mode == "symmetric" else 0),  # sign
            int(0),  # narrow range
            "ROUND",  # rounding mode
        )

    def __repr__(self):
        repr = (
            f"{self.__class__.__name__}(weight_bit={self.hawq_layer.weight_bit},"
            + f" bias_bit={self.hawq_layer.bias_bit}, quantize={self.hawq_layer.quant_mode})"
        )
        return repr

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]

        if self.export_mode:
            x = x / prev_act_scaling_factor.view(-1)[0]

            quant_node = get_quant_func(self.weight_node_inputs[2].item())
            weights = quant_node.apply(
                self.fc.weight.data, *self.weight_node_inputs, *self.node_attributes
            )
            x = torch.matmul(x, weights)

            if self.has_bias:
                quant_node = get_quant_func(self.bias_node_inputs[2].item())
                bias = quant_node.apply(
                    self.fc.bias.data, *self.bias_node_inputs, *self.node_attributes
                )
                x = torch.add(x, bias)

            # x = torch.round(x)
            bias_scaling_factor = self.scale * prev_act_scaling_factor.clone()
            if len(bias_scaling_factor) == 1:
                x = x * bias_scaling_factor.item()
            else:
                x = x * bias_scaling_factor
            model_info["dense_out"][self.hawq_layer] = x.clone()
            return x
        else:
            x = self.hawq_layer(x, prev_act_scaling_factor)
            return x


# ------------------------------------------------------------
class ExportQonnxQuantConv2d(nn.Module):
    def __init__(self, hawq_layer) -> None:
        super().__init__()
        self.hawq_layer = hawq_layer
        self.export_mode = False

        self.conv_scaling_factor = self.hawq_layer.conv_scaling_factor.detach().clone()
        self.weight_node_inputs = (
            torch.tensor(1, dtype=torch.float32),  # scale
            torch.tensor(0, dtype=torch.float32),  # zero point
            torch.tensor(self.hawq_layer.weight_bit, dtype=torch.float32),  # bit width
        )

        self.has_bias = False
        self.bias = None
        if self.hawq_layer.quantize_bias and (self.hawq_layer.bias is not None):
            self.has_bias = True
            self.bias_node_inputs = (
                torch.tensor(1, dtype=torch.float32),  # scale
                torch.tensor(0, dtype=torch.float32),  # zero point
                torch.tensor(self.hawq_layer.bias_bit, dtype=torch.float32),  # bit width
            )

        self.node_attributes = (
            int(1 if self.hawq_layer.quant_mode == "symmetric" else 0),  # sign
            int(0),  # narrow range
            "ROUND",  # rounding mode
        )

        dilation = self.hawq_layer.conv.dilation
        if type(dilation) != tuple:
            dilation = (dilation, dilation)

        kernel_size = self.hawq_layer.conv.kernel_size
        if type(kernel_size) != tuple:
            kernel_size = (kernel_size, kernel_size)

        pads = self.hawq_layer.conv.padding
        if type(pads) != tuple:
            pads = (pads, pads, pads, pads)
        elif len(pads) == 2:
            pads = (pads[0], pads[0], pads[1], pads[1])

        strides = self.hawq_layer.conv.stride
        if type(strides) != tuple:
            strides = (strides, strides)

        self.conv_args = (
            dilation,
            self.hawq_layer.conv.groups,
            kernel_size,
            pads,
            strides,
        )

    def __repr__(self):
        repr = f"{self.__class__.__name__}()"
        return repr

    def forward(self, x, prev_act_scaling_factor=None):
        if type(x) is tuple:
            prev_act_scaling_factor = x[1]
            x = x[0]
        if x.ndim == 3:
            # add an extra dimension for batch size
            x = x[None]

        bias_scaling_factor = self.conv_scaling_factor.view(
            1, -1
        ) * prev_act_scaling_factor.view(1, -1)
        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)
        correct_output_scale = correct_output_scale[0][0][0][0]

        if self.export_mode:
            QuantFunc = get_quant_func(self.hawq_layer.weight_bit)
            weights = QuantFunc.apply(self.hawq_layer.weight_integer.type(torch.float32), *self.weight_node_inputs, *self.node_attributes)
            x = x / prev_act_scaling_factor.item()
            if self.has_bias:
                quant_node = get_quant_func(self.bias_node_inputs[2].item())
                self.bias = quant_node.apply(
                    self.hawq_layer.bias_integer.data, *self.bias_node_inputs, *self.node_attributes
                )
            return (
                ConvFunc.apply(x, weights, self.bias, self.hawq_layer, *self.conv_args)
                * correct_output_scale.item(),
                self.conv_scaling_factor,
            )
        else:
            x, conv_scaling_factor = self.hawq_layer(x, prev_act_scaling_factor)
            return (x, conv_scaling_factor)


# ------------------------------------------------------------
class ExportQonnxQuantBnConv2d(nn.Module):
    def __init__(self, hawq_layer) -> None:
        super().__init__()
        self.hawq_layer = hawq_layer
        self.export_mode = False
        self.init_conv()

        self.fix_BN = self.hawq_layer.fix_BN
        self.bn = torch.nn.BatchNorm2d(
            self.hawq_layer.bn.num_features,
            self.hawq_layer.bn.eps,
            self.hawq_layer.bn.momentum,
        )

        self.bn.weight = self.hawq_layer.bn.weight
        self.bn.bias = self.hawq_layer.bn.bias
        self.bn.running_mean = self.hawq_layer.bn.running_mean
        self.bn.running_var = self.hawq_layer.bn.running_var

        self.output_factor = self.bn.weight.view(1, -1, 1, 1) / torch.sqrt(
            self.bn.running_var + self.bn.eps
        ).view(1, -1, 1, 1)

    def __repr__(self):
        s = f"{self.__class__.__name__}()"
        return s

    def init_conv(self):
        w_transform = self.hawq_layer.conv.weight.data.contiguous().view(
            self.hawq_layer.conv.out_channels, -1
        )
        w_min = w_transform.min(dim=1).values
        w_max = w_transform.max(dim=1).values
        self.conv_scaling_factor = symmetric_linear_quantization_params(
            self.hawq_layer.weight_bit, w_min, w_max, self.hawq_layer.per_channel
        )
        self.weight_integer = SymmetricQuantFunction.apply(
            self.hawq_layer.conv.weight,
            self.hawq_layer.weight_bit,
            self.conv_scaling_factor,
        )

        quant_layer = QuantConv2d()
        quant_layer.set_param(self.hawq_layer.conv)
        quant_layer.weight_integer = self.weight_integer
        quant_layer.conv_scaling_factor = self.conv_scaling_factor
        self.export_quant_conv = ExportQonnxQuantConv2d(quant_layer)

    def forward(self, x, pre_act_scaling_factor=None):
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]

        if self.export_mode:
            x, conv_scaling_factor = self.export_quant_conv(x, pre_act_scaling_factor)
            if self.fix_BN == False:
                return (
                    self.bn(x),
                    conv_scaling_factor.view(-1) * self.output_factor.view(-1),
                )
            else:
                return (
                    x, 
                    conv_scaling_factor.view(-1) * self.output_factor.view(-1),
                )
        else:
            x, convbn_scaling_factor = self.hawq_layer(x, pre_act_scaling_factor)
            return (x, convbn_scaling_factor)


# ------------------------------------------------------------
class ExportQonnxQuantAveragePool2d(nn.Module):
    def __init__(self, hawq_layer) -> None:
        super().__init__()
        self.hawq_layer = hawq_layer
        self.export_mode = False

        self.trunc_args = (
            torch.tensor(1),  # scale
            torch.tensor(0),  # zero point
            torch.tensor(32),  # input bit width
            torch.tensor(32),  # output bit width
            "ROUND",  # rounding mode
        )

    def __repr__(self):
        repr = f"{self.__class__.__name__}()"
        return repr

    def forward(self, x, x_scaling_factor=None):
        if type(x) is tuple:
            x_scaling_factor = x[1]
            x = x[0]

        if x_scaling_factor is None:
            return (self.hawq_layer(x), x_scaling_factor)

        if self.export_mode:
            x_scaling_factor = x_scaling_factor.view(-1)
            correct_scaling_factor = x_scaling_factor

            x_int = x / correct_scaling_factor
            x_int = RoundFunc.apply(x_int)
            x_int = self.hawq_layer.final_pool(x_int)

            x_int = TruncFunc.apply(x_int + 0.01, *self.trunc_args)

            return (x_int * correct_scaling_factor, correct_scaling_factor)
        else:
            return (self.hawq_layer(x), x_scaling_factor)
