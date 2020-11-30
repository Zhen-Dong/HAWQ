import tvm
from tvm import relay

from . import layers
from .init import create_workload, QuantizeInitializer



def residual_unit_v1(data,
                  input_channels,
                  num_filter,
                  stride,
                  dim_match,
                  name,
                  bottle_neck=True,
                  data_layout="NCHW",
                  kernel_layout="OIHW",
                  with_bn=False,
                  debug=None,
                  rounding='TRUNCATE'):
    """Return ResNet Unit symbol for building ResNet

    Parameters
    ----------
    data : str
        Input data

    input_channels: int
        Number of input channels

    num_filter : int
        Number of output channels

    bnf : int
        Bottle neck channels factor with regard to num_filter

    stride : tuple
        Stride used in convolution

    dim_match : bool
        True means channel number between input and output is the same,
        otherwise means differ

    name : str
        Base name of the operators
    """

    add_bias = not with_bn

    qconfig1 = layers.get_qconfig(name + '_qconfig1')

    req1 = layers.requantize(data,
                            input_scale=qconfig1.from_scale,
                            input_zero_point=qconfig1.from_zero_point,
                            output_scale=qconfig1.input_scale,
                            output_zero_point=qconfig1.input_zero_point,
                            out_dtype=qconfig1.input_dtype,
                            rounding=rounding)

    if debug:
        return req1

    if bottle_neck:

        conv1 = layers.quantized_conv2d(data=req1,
                                        name=name + '_qconv1',
                                        add_bias=add_bias,
                                        input_channels=input_channels,
                                        output_channels=int(num_filter*0.25),
                                        kernel_dtype=qconfig1.kernel_dtype,
                                        input_scale=qconfig1.input_scale,
                                        kernel_scale=qconfig1.kernel_scale,
                                        input_zero_point=qconfig1.input_zero_point,
                                        kernel_zero_point=qconfig1.kernel_zero_point,
                                        kernel_size=(1, 1),
                                        strides=stride, padding=(0, 0),
                                        data_layout=data_layout,
                                        kernel_layout=kernel_layout)

        if with_bn:
            conv1 = relay.cast(conv1, "int32")
            conv1 = layers.quantized_batchnorm(conv1, data_layout=data_layout, channels=int(num_filter*0.25), name = name + '_bn1')

        act1 = relay.nn.relu(data=conv1)

        qconfig2 = layers.get_qconfig(name + '_qconfig2')

        req2 = layers.requantize(data=act1,
                                 input_scale=qconfig1.output_scale,
                                 input_zero_point=0.0,
                                 output_scale=qconfig2.input_scale,
                                 output_zero_point=qconfig2.input_zero_point,
                                 out_dtype=qconfig2.input_dtype,
                                 rounding=rounding)

        conv2 = layers.quantized_conv2d(data=req2,
                                        name=name + '_qconv2',
                                        add_bias=add_bias,
                                        input_channels=int(num_filter*0.25),
                                        output_channels=int(num_filter*0.25),
                                        kernel_dtype=qconfig2.kernel_dtype,
                                        input_scale=qconfig2.input_scale,
                                        kernel_scale=qconfig2.kernel_scale,
                                        input_zero_point=qconfig2.input_zero_point,
                                        kernel_zero_point=qconfig2.kernel_zero_point,
                                        kernel_size=(3, 3),
                                        strides=(1,1), padding=(1, 1),
                                        data_layout=data_layout, kernel_layout=kernel_layout)

        if with_bn:
            conv2 = relay.cast(conv2, "int32")
            conv2 = layers.quantized_batchnorm(conv2, data_layout=data_layout, channels=int(num_filter*0.25), name = name + '_bn2')

        act2 = relay.nn.relu(data=conv2)

        qconfig3 = layers.get_qconfig(name + '_qconfig3')

        req3 = layers.requantize(act2,
                                 input_scale=qconfig2.output_scale,
                                 input_zero_point=0.0,
                                 output_scale=qconfig3.input_scale,
                                 output_zero_point=qconfig3.input_zero_point,
                                 out_dtype=qconfig3.input_dtype,
                                 rounding=rounding)

        conv3 = layers.quantized_conv2d(data=req3,
                                        name=name + '_qconv3',
                                        add_bias=add_bias,
                                        input_channels=int(num_filter*0.25),
                                        output_channels=num_filter,
                                        kernel_dtype=qconfig3.kernel_dtype,
                                        input_scale=qconfig3.input_scale,
                                        kernel_scale=qconfig3.kernel_scale,
                                        input_zero_point=qconfig3.input_zero_point,
                                        kernel_zero_point=qconfig3.kernel_zero_point,
                                        kernel_size=(1, 1),
                                        strides=(1, 1), padding=(0, 0),
                                        data_layout=data_layout, kernel_layout=kernel_layout)

        if dim_match:
            shortcut = data
        else:
            qconfig_sc = layers.get_qconfig(name + '_qconfig_sc')

            shortcut = layers.quantized_conv2d(data=req1,
                                               name=name + '_qsc',
                                               input_channels=input_channels,
                                               output_channels=num_filter,
                                               add_bias=add_bias,
                                               kernel_dtype=qconfig1.kernel_dtype,
                                               input_scale=qconfig1.input_scale,
                                               kernel_scale=qconfig_sc.kernel_scale,
                                               input_zero_point=qconfig1.input_zero_point,
                                               kernel_zero_point=qconfig_sc.kernel_zero_point,
                                               kernel_size=(1, 1),
                                               strides=stride,
                                               data_layout=data_layout, kernel_layout=kernel_layout)

        qconfig_add = layers.get_qconfig(name + '_qconfig_add')

        add1 = layers.add(lhs=conv3,
                          rhs=shortcut,
                          lhs_scale=qconfig3.output_scale,
                          lhs_zero_point=0.0,
                          rhs_scale=qconfig1.from_scale if dim_match else qconfig_sc.output_scale,
                          rhs_zero_point=0.0,
                          output_scale=qconfig_add.output_scale,
                          output_zero_point=0.0)

        if with_bn:
            add1 = relay.cast(add1, "int32")
            add1 = layers.quantized_batchnorm(add1, data_layout=data_layout, channels=num_filter, name = name + '_bn3')

        act3 = relay.nn.relu(data=add1)

        act3 = relay.annotation.stop_fusion(act3)

        return act3

    conv1 = layers.quantized_conv2d(data=req1,
                                    name=name + '_qconv1',
                                    input_channels=input_channels,
                                    output_channels=num_filter,
                                    add_bias=add_bias,
                                    kernel_dtype=qconfig1.kernel_dtype,
                                    input_scale=qconfig1.input_scale,
                                    kernel_scale=qconfig1.kernel_scale,
                                    input_zero_point=qconfig1.input_zero_point,
                                    kernel_zero_point=qconfig1.kernel_zero_point,
                                    kernel_size=(3, 3),
                                    strides=stride, padding=(1, 1),
                                    data_layout=data_layout, kernel_layout=kernel_layout)

    if with_bn:
        conv1 = relay.cast(conv1, "int32")
        conv1 = layers.quantized_batchnorm(conv1, data_layout=data_layout, channels=num_filter, name = name + '_bn1')

    act1 = relay.nn.relu(data=conv1)

    qconfig2 = layers.get_qconfig(name + '_qconfig2')

    req2 = layers.requantize(data=act1,
                             input_scale=qconfig2.from_scale,
                             input_zero_point=qconfig2.from_zero_point,
                             output_scale=qconfig2.input_scale,
                             output_zero_point=qconfig2.input_zero_point,
                             out_dtype=qconfig2.input_dtype,
                             rounding=rounding)

    conv2 = layers.quantized_conv2d(data=req2,
                                    name=name + '_qconv2',
                                    input_channels=num_filter,
                                    output_channels=num_filter,
                                    add_bias=add_bias,
                                    kernel_dtype=qconfig2.kernel_dtype,
                                    input_scale=qconfig2.input_scale,
                                    kernel_scale=qconfig2.kernel_scale,
                                    input_zero_point=qconfig2.input_zero_point,
                                    kernel_zero_point=qconfig2.kernel_zero_point,
                                    kernel_size=(3, 3),
                                    strides=(1, 1), padding=(1, 1),
                                    data_layout=data_layout, kernel_layout=kernel_layout)

    if dim_match:
        shortcut = data
    else:
        qconfig_sc = layers.get_qconfig(name + '_qconfig_sc')

        shortcut = layers.quantized_conv2d(data=req1,
                                           name=name + '_qsc',
                                           input_channels=input_channels,
                                           output_channels=num_filter,
                                           add_bias=add_bias,
                                           kernel_dtype=qconfig1.kernel_dtype,
                                           input_scale=qconfig1.input_scale,
                                           kernel_scale=qconfig_sc.kernel_scale,
                                           input_zero_point=qconfig1.input_zero_point,
                                           kernel_zero_point=qconfig_sc.kernel_zero_point,
                                           kernel_size=(1, 1),
                                           strides=stride,
                                           data_layout=data_layout, kernel_layout=kernel_layout)



    qconfig_add = layers.get_qconfig(name + '_qconfig_add')

    add1 = layers.add(lhs=conv2,
                      rhs=shortcut,
                      lhs_scale=qconfig2.output_scale,
                      lhs_zero_point=0.0,
                      rhs_scale=qconfig1.from_scale if dim_match else qconfig_sc.output_scale,
                      rhs_zero_point=qconfig1.from_zero_point if dim_match else 0.0,
                      output_scale=qconfig_add.output_scale,
                      output_zero_point=0.0)

    if with_bn:
        add1 = relay.cast(add1, "int32")
        add1 = layers.quantized_batchnorm(add1, data_layout=data_layout, channels=num_filter, name=name + '_bn2')

    act2 = relay.nn.relu(data=add1)
    act2 = relay.annotation.stop_fusion(act2)

    return act2


def qnn_resnet_v1(units,
           num_stages,
           filter_list,
           num_classes,
           data_shape,
           bottle_neck=True,
           data_layout="NCHW",
           kernel_layout="OIHW",
           dtype="int8",
           with_bn=False,
           debug_unit=None,
           rounding="TRUNCATE",
           with_softmax=True):
    """Return ResNet Program.

    Parameters
    ----------
    units : list
        Number of units in each stage

    num_stages : int
        Number of stages

    filter_list : list
        Channel size of each stage

    num_classes : int
        Ouput size of symbol

    data_shape : tuple of int.
        The shape of input data.

    bottle_neck : bool
        Whether apply bottleneck transformation.

    dtype : str
        The input data type.
    """
    num_unit = len(units)
    assert num_unit == num_stages

    add_bias = not with_bn

    data = relay.var("data", shape=data_shape, dtype=dtype)

    if data_layout == 'NCHW':
        (_, _, height, _) = data_shape
    elif data_layout == 'NHWC':
        (_, height, _, _) = data_shape
    elif data_layout == 'HWCN':
        (height, _, _, _) = data_shape
    elif data_layout == 'HWNC':
        data = relay.var("data", shape=(data_shape[2], data_shape[0], data_shape[1], data_shape[3]), dtype=dtype)
        (height, _, _, _) = data_shape
    else:
        raise RuntimeError("Unsupported data layout {}".format(data_layout))

    qconfig_conv0 = layers.get_qconfig("conv0_qconfig")

    if data_layout == 'HWNC':
        # data = relay.transpose(data, [2, 0, 1, 3])
        _data_layout = 'NHWC'
    elif data_layout == 'HWCN':
        data = relay.transpose(data, [3, 0, 1, 2])
        _data_layout = 'NHWC'
    else:
        _data_layout = data_layout

    if height < 32:            # such as cifar10
        body = layers.quantized_conv2d(data=data,
                                        name='conv0',
                                        add_bias=add_bias,
                                        input_channels=3,
                                        output_channels=filter_list[0],
                                        kernel_dtype=qconfig_conv0.kernel_dtype,
                                        input_scale=qconfig_conv0.input_scale,
                                        kernel_scale=qconfig_conv0.kernel_scale,
                                        input_zero_point=qconfig_conv0.input_zero_point,
                                        kernel_zero_point=qconfig_conv0.kernel_zero_point,
                                        kernel_size=(3, 3),
                                        strides=(1, 1), padding=(1, 1),
                                        data_layout=_data_layout, kernel_layout=kernel_layout)
    else:                       # often expected to be 224 such as imagenet

        body = layers.quantized_conv2d(data=data,
                                        name='conv0',
                                        add_bias=add_bias,
                                        input_channels=3,
                                        output_channels=filter_list[0],
                                        kernel_dtype=qconfig_conv0.kernel_dtype,
                                        input_scale=qconfig_conv0.input_scale,
                                        kernel_scale=qconfig_conv0.kernel_scale,
                                        input_zero_point=qconfig_conv0.input_zero_point,
                                        kernel_zero_point=qconfig_conv0.kernel_zero_point,
                                        kernel_size=(7, 7),
                                        strides=(2, 2), padding=(3, 3),
                                        data_layout=_data_layout, kernel_layout=kernel_layout)

        if with_bn:
            body = relay.cast(body, "int32")
            body = layers.quantized_batchnorm(body, data_layout=_data_layout, channels=filter_list[0], name='bn0')

        body = layers.requantize(body,
                                 input_scale=qconfig_conv0.input_scale * qconfig_conv0.kernel_scale,
                                 output_scale=qconfig_conv0.output_scale,
                                 out_dtype='int32',
                                 rounding=rounding)

        body = relay.nn.relu(data=body)
        # Remove pooling for cifar10
        if num_classes == 10:
            body = relay.annotation.stop_fusion(body)
        else:
            body = relay.nn.max_pool2d(data=body, pool_size=(3, 3), strides=(2, 2), padding=(1, 1), layout=_data_layout)

        if with_bn:
            body = relay.cast(body, "int32")
            body = layers.quantized_batchnorm(body, data_layout=_data_layout, channels=filter_list[0], name='bn1')

    if data_layout == 'HWCN':
        body = relay.transpose(body, [1, 2, 3, 0])
    if data_layout == 'HWNC':
        body = relay.transpose(body, [1, 2, 0, 3])

    for i in range(num_stages):

        unit_input_name = "stage%d_unit1_input" % (i + 1)
        unit_output_name = "stage%d_unit1_output" % (i + 1)

        body = residual_unit_v1(
            body, filter_list[i], filter_list[i+1], (1 if i == 0 else 2, 1 if i == 0 else 2),
            (filter_list[i] == filter_list[i+1]), name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck,
            data_layout=data_layout, kernel_layout=kernel_layout, with_bn=with_bn, debug=(debug_unit==unit_input_name),
            rounding=rounding)

        if debug_unit == unit_input_name or debug_unit == unit_output_name:
            return body

        for j in range(units[i]-1):

            unit_input_name = "stage%d_unit%d_input" % (i + 1, j + 2)
            unit_output_name = "stage%d_unit%d_output" % (i + 1, j + 2)


            body = residual_unit_v1(
                body, filter_list[i+1], filter_list[i+1], (1, 1), True,
                name='stage%d_unit%d' % (i + 1, j + 2), bottle_neck=bottle_neck,
                data_layout=data_layout, kernel_layout=kernel_layout, with_bn=with_bn, debug=(debug_unit==unit_input_name),
                rounding=rounding)

            if debug_unit == unit_input_name or debug_unit == unit_output_name:
                    return body

    # Tranpose back to NHWC because pooling and dense don't support HWNC
    if data_layout == 'HWCN':
        body = relay.transpose(body, [1, 2, 3, 0])
    if data_layout == 'HWNC':
        body = relay.transpose(body, [2, 0, 1, 3])

    # Remove pooling for cifar10
    if num_classes != 10:
        # Although kernel is not used here when global_pool=True, we should put one
        body = relay.nn.global_avg_pool2d(data=body, layout=_data_layout)

    if debug_unit == "avg_pool":
        return body

    qconfig_fc = layers.get_qconfig(name='fc_qconfig')
    req1 = layers.requantize(body,
                            input_scale=qconfig_fc.from_scale,
                            input_zero_point=qconfig_fc.from_zero_point,
                            output_scale=qconfig_fc.input_scale,
                            output_zero_point=qconfig_fc.input_zero_point,
                            out_dtype='int8',
                            rounding=rounding)

    flat = relay.nn.batch_flatten(data=req1)

    if debug_unit == "fc_input":
        return flat

    if num_classes == 10:
        kernel_size = 16384
    else:
        kernel_size = filter_list[-1]

    fc = layers.quantized_dense(data=flat, name='fc',
                                input_zero_point= qconfig_fc.input_zero_point,
                                kernel_zero_point=qconfig_fc.kernel_zero_point,
                                input_scale=qconfig_fc.input_scale,
                                kernel_scale=qconfig_fc.kernel_scale,
                                units=num_classes,
                                kernel_shape=(num_classes, kernel_size),
                                kernel_dtype='int8',
                                add_bias=add_bias)

    if debug_unit == "fc_output":
        return fc

    net = layers.dequantize(fc, input_scale=qconfig_fc.output_scale, input_zero_point=0.0)

    if with_softmax:
        net = relay.nn.softmax(data=net)

    return relay.Function(relay.analysis.free_vars(net), net)


def get_net(batch_size,
            num_classes,
            num_layers=50,
            image_shape=(3, 224, 224),
            data_layout="NCHW",
            kernel_layout="OIHW",
            dtype="float32",
            with_bn=False,
            debug_unit=None,
            rounding="TRUNCATE",
            with_softmax=True,
            **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    (_, height, _) = image_shape

    if data_layout == 'NCHW':
        data_shape = (batch_size,) + image_shape
    elif data_layout == 'NHWC':
        data_shape = (batch_size, image_shape[1], image_shape[2], image_shape[0])
    elif data_layout == 'HWCN':
        data_shape = (image_shape[1], image_shape[2], image_shape[0], batch_size)
    elif data_layout == 'HWNC':
        data_shape = (image_shape[1], image_shape[2], batch_size, image_shape[0])
    else:
        raise RuntimeError("Unsupported data layout {}".format(data_layout))

    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}".format(num_layers))
        # Remove stage 4 for cifar10
        if num_classes == 10:
            num_stages = 3
            units.pop()
            filter_list.pop()

    return qnn_resnet_v1(units=units,
                  num_stages=num_stages,
                  filter_list=filter_list,
                  num_classes=num_classes,
                  data_shape=data_shape,
                  bottle_neck=bottle_neck,
                  data_layout=data_layout,
                  kernel_layout=kernel_layout,
                  dtype=dtype,
                  with_bn=with_bn,
                  debug_unit=debug_unit,
                  rounding=rounding,
                  with_softmax=with_softmax)



def get_workload(batch_size=1,
                 num_classes=1000,
                 num_layers=18,
                 image_shape=(3, 224, 224),
                 dtype="float32",
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 with_bn=False,
                 debug_unit=None,
                 rounding="TONEAREST",
                 with_softmax=True,
                 **kwargs):
    """Get benchmark workload for resnet

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    num_layers : int, optional
        Number of layers

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    kwargs : dict
        Extra arguments

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a ResNet network.

    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size=batch_size,
                  num_classes=num_classes,
                  num_layers=num_layers,
                  image_shape=image_shape,
                  dtype=dtype,
                  data_layout=data_layout,
                  kernel_layout=kernel_layout,
                  with_bn=with_bn,
                  debug_unit=debug_unit,
                  rounding=rounding,
                  with_softmax=with_softmax,
                  **kwargs)

    return create_workload(net, QuantizeInitializer())

