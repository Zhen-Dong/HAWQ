"""
    Quantized MobileNetV2 for ImageNet-1K, implemented in PyTorch.
    Original paper: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
"""

import os
import torch.nn as nn
import torch.nn.init as init
from ..quantization_utils.quant_modules import *


class Q_LinearBottleneck(nn.Module):
    def __init__(self,
                model,
                in_channels,
                out_channels,
                stride,
                expansion,
                remove_exp_conv):
        """
        So-called 'Linear Bottleneck' layer. It is used as a quantized MobileNetV2 unit.
        Parameters:
        ----------
        model : nn.Module
            The pretrained floating-point couterpart of this module with the same structure.
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        stride : int or tuple/list of 2 int
            Strides of the second convolution layer.
        expansion : bool
            Whether do expansion of channels.
        remove_exp_conv : bool
            Whether to remove expansion convolution.
        """
        super(Q_LinearBottleneck, self).__init__()
        self.residual = (in_channels == out_channels) and (stride == 1)
        mid_channels = in_channels * 6 if expansion else in_channels
        self.use_exp_conv = (expansion or (not remove_exp_conv))
        self.activatition_func = nn.ReLU6()

        self.quant_act = QuantAct()

        if self.use_exp_conv:
            self.conv1 = QuantBnConv2d()
            self.conv1.set_param(model.conv1.conv, model.conv1.bn)
            self.quant_act1 = QuantAct()

        self.conv2 = QuantBnConv2d()
        self.conv2.set_param(model.conv2.conv, model.conv2.bn)
        self.quant_act2 = QuantAct()

        self.conv3 = QuantBnConv2d()
        self.conv3.set_param(model.conv3.conv, model.conv3.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        if self.residual:
            identity = x

        x, act_scaling_factor = self.quant_act(x, scaling_factor_int32, None, None, None, None)

        if self.use_exp_conv:
            x, weight_scaling_factor = self.conv1(x, act_scaling_factor)
            x = self.activatition_func(x)
            x, self.act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor, None, None)

            x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
            x = self.activatition_func(x)
            x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

            # note that, there is no activation for the last conv
            x, weight_scaling_factor = self.conv3(x, act_scaling_factor)
        else:
            x, weight_scaling_factor = self.conv2(x, act_scaling_factor)
            x = self.activatition_func(x)
            x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor, None, None)

            # note that, there is no activation for the last conv
            x, weight_scaling_factor = self.conv3(x, act_scaling_factor)

        if self.residual:
            x = x + identity
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None, None)

        return x, act_scaling_factor


class Q_MobileNetV2(nn.Module):
    """
    Quantized MobileNetV2 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,' https://arxiv.org/abs/1801.04381.
    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point MobileNetV2.
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    remove_exp_conv : bool
        Whether to remove expansion convolution.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 model,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 remove_exp_conv,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(Q_MobileNetV2, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.channels = channels
        self.activatition_func = nn.ReLU6()

        # add input quantization
        self.quant_input = QuantAct()

        # change the inital block
        self.add_module("init_block", QuantBnConv2d())

        self.init_block.set_param(model.features.init_block.conv, model.features.init_block.bn)

        self.quant_act_int32 = QuantAct()

        self.features = nn.Sequential()
        # change the middle blocks
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            cur_stage = getattr(model.features, f'stage{i+1}')
            for j, out_channels in enumerate(channels_per_stage):
                cur_unit = getattr(cur_stage, f'unit{j+1}')

                stride = 2 if (j == 0) and (i != 0) else 1
                expansion = (i != 0) or (j != 0)

                stage.add_module("unit{}".format(j + 1), Q_LinearBottleneck(
                    cur_unit,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion,
                    remove_exp_conv=remove_exp_conv,
                    ))

                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        # change the final block
        self.quant_act_before_final_block = QuantAct()
        self.features.add_module("final_block", QuantBnConv2d())

        self.features.final_block.set_param(model.features.final_block.conv, model.features.final_block.bn)
        self.quant_act_int32_final = QuantAct()

        in_channels = final_block_channels

        self.features.add_module("final_pool", QuantAveragePool2d())
        self.features.final_pool.set_param(model.features.final_pool)
        self.quant_act_output = QuantAct()

        self.output = QuantConv2d()
        self.output.set_param(model.output)

    def forward(self, x):
        # quantize input
        x, act_scaling_factor = self.quant_input(x)

        # the init block
        x, weight_scaling_factor = self.init_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None)

        # the feature block
        for i, channels_per_stage in enumerate(self.channels):
            cur_stage = getattr(self.features, f'stage{i+1}')
            for j, out_channels in enumerate(channels_per_stage):
                cur_unit = getattr(cur_stage, f'unit{j+1}')

                x, act_scaling_factor = cur_unit(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act_before_final_block(x, act_scaling_factor, None, None, None, None)
        x, weight_scaling_factor = self.features.final_block(x, act_scaling_factor)
        x = self.activatition_func(x)
        x, act_scaling_factor = self.quant_act_int32_final(x, act_scaling_factor, weight_scaling_factor, None, None, None)

        # the final pooling
        x = self.features.final_pool(x, act_scaling_factor)

        # the output
        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor, None, None, None, None)
        x, act_scaling_factor = self.output(x, act_scaling_factor)

        x = x.view(x.size(0), -1)

        return x


def q_get_mobilenetv2(model, width_scale, remove_exp_conv=False):
    """
    Create quantized MobileNetV2 model with specific parameters.
    Parameters:
    ----------
    model : nn.Module
        The pretrained floating-point MobileNetV2.
    width_scale : float
        Scale factor for width of layers.
    remove_exp_conv : bool, default False
        Whether to remove expansion convolution.
    """

    init_block_channels = 32
    final_block_channels = 1280
    layers = [1, 2, 3, 4, 3, 3, 1]
    downsample = [0, 1, 1, 1, 0, 1, 0]
    channels_per_layers = [16, 24, 32, 64, 96, 160, 320]

    from functools import reduce
    channels = reduce(
        lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
        zip(channels_per_layers, layers, downsample),
        [[]])

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = int(final_block_channels * width_scale)

    net = Q_MobileNetV2(
        model,
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        remove_exp_conv=remove_exp_conv)

    return net


def q_mobilenetv2_w1(model):
    """
    Quantized 1.0 MobileNetV2-224 model from 'MobileNetV2: Inverted Residuals and Linear Bottlenecks,'
    https://arxiv.org/abs/1801.04381.
    Parameters:
    model : nn.Module
        The pretrained floating-point MobileNetV2.
    """
    return q_get_mobilenetv2(model, width_scale=1.0)
