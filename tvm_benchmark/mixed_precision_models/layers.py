"""Simple Layer DSL wrapper to ease creation of neural nets."""
from tvm import relay
from collections import namedtuple
from topi.util import get_const_tuple

import numpy as np

QConfig = namedtuple('QConfig', 'from_dtype, from_scale, from_zero_point, \
                                input_dtype, input_scale, input_zero_point, \
                                kernel_dtype, kernel_scale, kernel_zero_point, \
                                output_dtype, output_scale, output_zero_point',
                      defaults=('int32', 65.0, 0.0, 'int8', 8.0, 0.0, 'int8', 8.0, 0.0, 'int32', 74.0, 0.0))

class QuantizeContext(object):
    qconfig_dict = dict()
    qconfig_print = dict()
    default_qconfig = QConfig()

    @staticmethod
    def read_qconfig_from_file(file_path):
        pass

    @staticmethod
    def set_default_qconfig(qconfig):
        QuantizeContext.default_qconfig = qconfig

def get_qconfig(name):
    if name in QuantizeContext.qconfig_dict:
        return QuantizeContext.qconfig_dict[name]
    else:
        QuantizeContext.qconfig_print[name] = QuantizeContext.default_qconfig
        return QuantizeContext.default_qconfig


def quantized_conv2d(data,
                     kernel_dtype,
                     name,
                     input_channels,
                     kernel_size,
                     output_channels,
                     strides=(1, 1),
                     padding=(0, 0),
                     weight=None,
                     add_bias=False,
                     input_scale=8.0,
                     kernel_scale=8.0,
                     input_zero_point=0.0,
                     kernel_zero_point=0.0,
                     data_layout='NCHW',
                     kernel_layout='OIHW',
                     **kwargs):

    """Wrapper of qnn.conv2d
    Parameters
    ----------
    data : relay.Expr
        The input expression.

    weight : relay.Expr
        The weight to conv2d.

    name : str
        The name of this convolution.

    input_channels: int
        The number of input channels.

    out_channels: int
        The number of output channels.

    input_scale : float
        The scale of input.

    kernel_scale : float
        The scale of kernel.

    input_zero_point : float
        The zero point of input.

    kernel_zero_point : float
        The zero point of kernel.

    kwargs : dict
        Additional arguments.

    Returns
    -------
    result : relay.Expr
        The result.
    """

    # print("%s, %s, %d, %d, %d, %d, %d" % (, kernel_dtype, input_channels, output_channels, kernel_size[0], strides[0], padding[0]))

    input_zero_point = relay.const(input_zero_point, 'int32')
    kernel_zero_point = relay.const(kernel_zero_point, 'int32')

    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(input_scale.astype('float32'), 'float32')

    if isinstance(kernel_scale, float):
        kernel_scale = relay.const(kernel_scale, 'float32')
    else:
        kernel_scale = relay.const(kernel_scale.astype('float32'), 'float32')

    if kernel_layout == "OIHW":
        kernel_shape = (output_channels, input_channels, kernel_size[0], kernel_size[1])
    elif kernel_layout == "HWIO":
        kernel_shape = (kernel_size[0], kernel_size[1], input_channels, output_channels)
    elif kernel_layout == "HWOI":
        kernel_shape = (kernel_size[0], kernel_size[1], output_channels, input_channels)
    elif kernel_layout == "OHWI":
        kernel_shape = (output_channels, kernel_size[0], kernel_size[1], input_channels)
    else:
        raise RuntimeError("Unsupported kernel layout {}".format(kernel_layout))

    if weight is None:
        weight = relay.var(name + "_weight", shape=kernel_shape, dtype=kernel_dtype)

    conv2d = relay.qnn.op.conv2d(data, weight, input_zero_point, kernel_zero_point, input_scale, kernel_scale,
                                kernel_size=kernel_size, channels=output_channels, data_layout=data_layout, kernel_layout=kernel_layout, strides=strides, padding=padding, **kwargs)

    if add_bias:
        if data_layout == 'NCHW':
            bias_shape = (1, output_channels, 1, 1)
        elif data_layout == 'NHWC':
            bias_shape = (1, 1, 1, output_channels)
        elif data_layout == 'HWCN':
            bias_shape = (1, 1, output_channels, 1)
        elif data_layout == 'HWNC':
            bias_shape = (1, 1, 1, output_channels)
        else:
            raise RuntimeError("Unsupported conv2d layout {}".format(data_layout))

        bias = relay.var(name + "_bias", shape=bias_shape, dtype="int32")
        return relay.add(conv2d, bias)
    else:
        return conv2d


def quantized_batchnorm(data, data_layout, channels, name, dtype='int32'):
    if data_layout == 'NCHW':
        bn_shape = (1, channels, 1, 1)
    elif data_layout == 'NHWC':
        bn_shape = (1, 1, 1, channels)
    else:
        raise RuntimeError("Unsupported conv2d layout {}".format(data_layout))

    bn_mul = relay.var(name + '_scale', shape=bn_shape, dtype=dtype)
    bn_add = relay.var(name + '_shift', shape=bn_shape, dtype=dtype)

    mul = relay.multiply(data, bn_mul)
    add = relay.add(mul, bn_add)

    return add


def quantize(data,
             output_scale=8.0,
             output_zero_point=0.0,
             axis=-1,
             out_dtype='int8'):

    output_scale = relay.const(output_scale, 'float32')
    output_zero_point = relay.const(output_zero_point, 'int32')

    return relay.qnn.op.quantize(data, output_scale, output_zero_point, axis, out_dtype)


def requantize(data,
               input_scale=8.0,
               input_zero_point=0.0,
               output_scale=8.0,
               output_zero_point=0.0,
               axis=-1,
               rounding="TRUNCATE",
               out_dtype="int8"):

    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(np.array(input_scale).astype('float32'))

    input_zero_point = relay.const(input_zero_point, 'int32')

    if isinstance(output_scale, float):
        output_scale = relay.const(output_scale, 'float32')
    else:
        output_scale = relay.const(np.array(output_scale).astype('float32'))

    output_zero_point = relay.const(output_zero_point, 'int32')

    return relay.qnn.op.requantize(data,
                                   input_scale,
                                   input_zero_point,
                                   output_scale,
                                   output_zero_point,
                                   axis,
                                   rounding,
                                   out_dtype)


def dequantize(data,
               input_scale,
               input_zero_point,
               axis=-1):

    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(input_scale.astype('float32'), 'float32')

    input_zero_point = relay.const(input_zero_point, 'int32')

    return relay.qnn.op.dequantize(data,
               input_scale,
               input_zero_point,
               axis)


def add(lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point):

    if (np.ndim(lhs_scale) == 0 and np.ndim(rhs_scale) != 0) or \
       (np.ndim(lhs_scale) != 0 and np.ndim(rhs_scale) == 0) :
        if np.ndim(lhs_scale) == 0:
            lhs_scale = np.full_like(rhs_scale, lhs_scale)
        else:
            rhs_scale = np.full_like(lhs_scale, rhs_scale)

    lhs_scale = relay.const(lhs_scale, 'float32')
    lhs_zero_point = relay.const(lhs_zero_point, 'int32')

    rhs_scale = relay.const(rhs_scale, 'float32')
    rhs_zero_point = relay.const(rhs_zero_point, 'int32')

    if np.ndim(output_scale) == 1:
        output_scale = output_scale[0]
    if np.ndim(output_zero_point) == 1:
        output_zero_point = output_zero_point[0]

    output_scale = relay.const(output_scale, 'float32')
    output_zero_point = relay.const(output_zero_point, 'int32')

    return relay.qnn.op.add(lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point)


def quantized_dense(data,
          name,
          input_zero_point,
          kernel_zero_point,
          input_scale,
          kernel_scale,
          units,
          kernel_shape,
          kernel_dtype,
          add_bias=False,
          out_dtype="int32"):
    """Qnn Dense operator.
    Applies a quantized linear transformation

     .. math::

     `Y = X * W`

    Parameters
    ----------
    data : tvm.relay.Expr
        The quantized input data to the operator.
    weight : tvm.relay.Expr
        The quantized weight expressions.
    input_zero_point: tvm.relay.Expr
        The input zero point.
    kernel_zero_point: tvm.relay.Expr
        The kernel zero point.
    input_scale: tvm.relay.Expr
        The scale for the input tensor.
    kernel_scale: tvm.relay.Expr
        The scale for the weight tensor. The scale for the weight tensor is
        stored for access to this during relay. This information is not
        needed in the pass pipeline after qnn.conv2d is lowered to the
        sequence of steps as in nn.conv2d. See also input_scale in Requantize.
    units : int
        Number of hidden units of the dense transformation.
    out_dtype : str, optional
        Specifies the output data type for mixed precision dense can be int32 or int16.

    Returns
    -------
    result : tvm.relay.Expr
        The computed result.
    """

    input_zero_point = relay.const(input_zero_point, 'int32')
    kernel_zero_point = relay.const(kernel_zero_point, 'int32')
    if isinstance(input_scale, float):
        input_scale = relay.const(input_scale, 'float32')
    else:
        input_scale = relay.const(input_scale.astype('float32'), 'float32')

    if isinstance(kernel_scale, float):
        kernel_scale = relay.const(kernel_scale, 'float32')
    else:
        kernel_scale = relay.const(kernel_scale.astype('float32'), 'float32')

    weight = relay.var(name + "_weight", shape=kernel_shape, dtype=kernel_dtype)

    dense = relay.qnn.op.dense(data,
                               weight,
                               input_zero_point,
                               kernel_zero_point,
                               input_scale,
                               kernel_scale,
                               units,
                               out_dtype)
    if add_bias:
        bias = relay.var(name + "_bias", dtype="int32")
        return relay.nn.bias_add(dense, bias, axis=-1)
    else:
        return dense



def quantized_concatenate(data,
                input_scales,
                input_zero_points,
                output_scale,
                output_zero_point,
                axis):
    """Concatenate the quantized input tensors along the given axis.

    Parameters
    ----------
    data : Union(List[relay.Expr], Tuple[relay.Expr])
        The list of quantized tensors.

    input_scales : List[relay.Expr]
        The list of scales of input quantized tensors.

    input_zero_points : List[relay.Expr]
        The list of zero points of input quantized tensors.

    output_scale : relay.Expr
        The scale of the output quantized tensor.

    output_zero_point : relay.Expr
        The zero point of the output quantized tensor.

    axis : int
        The axis along which the tensors are concatenated.

    Returns
    -------
    result: relay.Expr
        The concatenated quantized tensor.
    """

    input_scales_expr = [relay.const(input_scale, 'float32') for input_scale in input_scales]
    input_zero_points_expr = [relay.const(input_zero_point, 'int32') for input_zero_point in input_zero_points]

    output_scale = relay.const(output_scale, 'float32')
    output_zero_point = relay.const(output_zero_point, 'int32')

    return relay.qnn.op.concatenate(data,
                input_scales_expr,
                input_zero_points_expr,
                output_scale,
                output_zero_point,
                axis)


def conv2d(data, in_dtype='int8', out_dtype='int32', weight=None, data_layout='NCHW', kernel_layout='OIHW', **kwargs):
    """Wrapper of conv2d which automatically creates weights if not given.
    Parameters
    ----------
    data : relay.Expr
        The input expression.
    weight : relay.Expr
        The weight to conv2d.
    kwargs : dict
        Additional arguments.
    Returns
    -------
    result : relay.Expr
        The result.
    """
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight", dtype=in_dtype)

    # Small hack here since conv2d_nhwc only supports kernel HWIO
    kernel_layout = 'HWIO' if data_layout == 'NHWC' and in_dtype == 'float32' else kernel_layout
    return relay.nn.conv2d(data, weight, out_dtype=out_dtype, data_layout=data_layout, kernel_layout=kernel_layout, **kwargs)


def dense_add_bias(data, weight=None, bias=None, units=None, **kwargs):
    """Wrapper of dense which automatically creates weights if not given.
    Parameters
    ----------
    data : relay.Expr
        The input expression.
    weight : relay.Expr
        The weight to conv2d.
    bias : relay.Expr
        The bias.
    kwargs : dict
        Additional arguments.
    Returns
    -------
    result : relay.Expr
        The result.
    """
    name = kwargs.get("name")
    kwargs.pop("name")
    if not weight:
        weight = relay.var(name + "_weight")
    if not bias:
        bias = relay.var(name + "_bias")
    data = relay.nn.dense(data, weight, units, **kwargs)
    data = relay.nn.bias_add(data, bias, axis=-1)
    return data