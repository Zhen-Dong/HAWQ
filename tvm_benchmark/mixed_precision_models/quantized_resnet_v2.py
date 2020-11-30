from tvm import relay

import layers
from init import create_workload, QuantizeInitializer

def residual_unit(data,
                  input_channels,
                  num_filter,
                  stride,
                  dim_match,
                  name,
                  bottle_neck=True,
                  data_layout="NCHW",
                  kernel_layout="OIHW",
                  ):
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

    if bottle_neck:
        bn1 = layers.quantized_batchnorm(data=data, data_layout=data_layout, channels=input_channels, name=name+"_qbn1")

        act1 = relay.nn.relu(data=bn1)
        act1 = relay.annotation.stop_fusion(act1)

        qconfig1 = layers.get_qconfig(name + '_qconfig1')

        req1 = layers.requantize(data=act1,
                                 input_scale=qconfig1.from_scale,
                                 input_zero_point=qconfig1.from_zero_point,
                                 output_scale=qconfig1.input_scale,
                                 output_zero_point=qconfig1.input_zero_point,
                                 out_dtype=qconfig1.input_dtype)

        conv1 = layers.quantized_conv2d(data=req1,
                                        name=name + '_qconv1',
                                        add_bias=True,
                                        input_channels=input_channels,
                                        output_channels=int(num_filter*0.25),
                                        kernel_dtype=qconfig1.kernel_dtype,
                                        input_scale=qconfig1.input_scale,
                                        kernel_scale=qconfig1.kernel_scale,
                                        input_zero_point=qconfig1.input_zero_point,
                                        kernel_zero_point=qconfig1.kernel_zero_point,
                                        kernel_size=(1, 1),
                                        strides=stride, padding=(0, 0),
                                        data_layout=data_layout, kernel_layout=kernel_layout)

        act2 = relay.nn.relu(data=conv1)
        act2 = relay.annotation.stop_fusion(act2)

        qconfig2 = layers.get_qconfig(name + '_qconfig2')

        req2 = layers.requantize(data=act2,
                                 input_scale=qconfig2.from_scale,
                                 input_zero_point=qconfig2.from_zero_point,
                                 output_scale=qconfig2.input_scale,
                                 output_zero_point=qconfig2.input_zero_point,
                                 out_dtype=qconfig2.input_dtype)

        conv2 = layers.quantized_conv2d(data=req2,
                                        name=name + '_qconv2',
                                        add_bias=True,
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

        act3 = relay.nn.relu(data=conv2)
        act3 = relay.annotation.stop_fusion(act3)

        qconfig3 = layers.get_qconfig(name + '_qconfig3')

        req3 = layers.requantize(act3,
                                 input_scale=qconfig3.from_scale,
                                 input_zero_point=qconfig3.from_zero_point,
                                 output_scale=qconfig3.input_scale,
                                 output_zero_point=qconfig3.input_zero_point,
                                 out_dtype=qconfig3.input_dtype)

        conv3 = layers.quantized_conv2d(data=req3,
                                        name=name + '_qconv3',
                                        add_bias=True,
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

            req_sc = layers.requantize(data=data,
                                       input_scale=qconfig_sc.from_scale,
                                       input_zero_point=qconfig_sc.from_zero_point,
                                       output_scale=qconfig_sc.input_scale,
                                       output_zero_point=qconfig_sc.input_zero_point,
                                       out_dtype=qconfig_sc.input_dtype)

            shortcut = layers.quantized_conv2d(data=req_sc,
                                               name=name+'_qsc',
                                               input_channels=input_channels,
                                               output_channels=num_filter,
                                               add_bias=False,
                                               kernel_dtype=qconfig_sc.kernel_dtype,
                                               input_scale=qconfig_sc.input_scale,
                                               kernel_scale=qconfig_sc.kernel_scale,
                                               input_zero_point=qconfig_sc.input_zero_point,
                                               kernel_zero_point=qconfig_sc.kernel_zero_point,
                                               kernel_size=(1, 1),
                                               strides=stride,
                                               data_layout=data_layout, kernel_layout=kernel_layout)

        qconfig_add = layers.get_qconfig(name + '_qconfig_add')

        shortcut = relay.annotation.stop_fusion(shortcut)

        return layers.add(lhs=conv3,
                          rhs=shortcut,
                          lhs_scale=qconfig3.input_scale * qconfig3.kernel_scale,
                          lhs_zero_point=0.0,
                          rhs_scale=qconfig_add.input_scale,
                          rhs_zero_point=qconfig_add.input_zero_point,
                          output_scale=qconfig_add.output_scale,
                          output_zero_point=qconfig_add.output_zero_point)

    bn1 = layers.quantized_batchnorm(data=data, data_layout=data_layout, channels=input_channels, name=name+"_qbn1" )

    act1 = relay.nn.relu(data=bn1)
    act1 = relay.annotation.stop_fusion(act1)

    qconfig1 = layers.get_qconfig(name + '_qconfig1')

    req1 = layers.requantize(data=act1,
                             input_scale=qconfig1.from_scale,
                             input_zero_point=qconfig1.from_zero_point,
                             output_scale=qconfig1.input_scale,
                             output_zero_point=qconfig1.input_zero_point,
                             out_dtype=qconfig1.input_dtype)

    conv1 = layers.quantized_conv2d(data=req1,
                                    name=name + '_qconv1',
                                    input_channels=input_channels,
                                    output_channels=num_filter,
                                    add_bias=True,
                                    kernel_dtype=qconfig1.kernel_dtype,
                                    input_scale=qconfig1.input_scale,
                                    kernel_scale=qconfig1.kernel_scale,
                                    input_zero_point=qconfig1.input_zero_point,
                                    kernel_zero_point=qconfig1.kernel_zero_point,
                                    kernel_size=(3, 3),
                                    strides=stride, padding=(1, 1),
                                    data_layout=data_layout, kernel_layout=kernel_layout)

    act2 = relay.nn.relu(data=conv1)
    act2 = relay.annotation.stop_fusion(act2)

    qconfig2 = layers.get_qconfig(name + '_qconfig2')

    req2 = layers.requantize(data=act2,
                             input_scale=qconfig2.from_scale,
                             input_zero_point=qconfig2.from_zero_point,
                             output_scale=qconfig2.input_scale,
                             output_zero_point=qconfig2.input_zero_point,
                             out_dtype=qconfig2.input_dtype)

    conv2 = layers.quantized_conv2d(data=req2,
                                    name=name + '_qconv2',
                                    input_channels=num_filter,
                                    output_channels=num_filter,
                                    add_bias=True,
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

        req_sc = layers.requantize(data=data,
                                 input_scale=qconfig_sc.from_scale,
                                 input_zero_point=qconfig_sc.from_zero_point,
                                 output_scale=qconfig_sc.input_scale,
                                 output_zero_point=qconfig_sc.input_zero_point,
                                 out_dtype=qconfig_sc.input_dtype)

        shortcut = layers.quantized_conv2d(data=req_sc,
                                           name=name+'_qsc',
                                           input_channels=input_channels,
                                           output_channels=num_filter,
                                           add_bias=False,
                                           kernel_dtype=qconfig_sc.kernel_dtype,
                                           input_scale=qconfig_sc.input_scale,
                                           kernel_scale=qconfig_sc.kernel_scale,
                                           input_zero_point=qconfig_sc.input_zero_point,
                                           kernel_zero_point=qconfig_sc.kernel_zero_point,
                                           kernel_size=(1, 1),
                                           strides=stride,
                                           data_layout=data_layout, kernel_layout=kernel_layout)

    qconfig_add = layers.get_qconfig(name + '_qconfig_add')

    shortcut = relay.annotation.stop_fusion(shortcut)

    return layers.add(lhs=conv2,
                      rhs=shortcut,
                      lhs_scale=qconfig2.input_scale * qconfig2.kernel_scale,
                      lhs_zero_point=0.0,
                      rhs_scale=qconfig_add.input_scale,
                      rhs_zero_point=qconfig_add.input_zero_point,
                      output_scale=qconfig_add.output_scale,
                      output_zero_point=qconfig_add.output_zero_point)

def qnn_resnet(units,
           num_stages,
           filter_list,
           num_classes,
           data_shape,
           bottle_neck=True,
           data_layout="NCHW",
           kernel_layout="OIHW",
           dtype="int8"):
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

    data = relay.var("data", shape=data_shape, dtype=dtype)

    if data_layout == 'NCHW':
        (_, _, height, _) = data_shape
    elif data_layout == 'NHWC':
        (_, height, _, _) = data_shape
    else:
        raise RuntimeError("Unsupported data layout {}".format(data_layout))

    if height <= 32:            # such as cifar10
        body = layers.conv2d(
            data=data, in_dtype=dtype, channels=filter_list[0], kernel_size=(3, 3),
            strides=(1, 1), padding=(1, 1), data_layout=data_layout, kernel_layout=kernel_layout, name="conv0")
    else:                       # often expected to be 224 such as imagenet
        body = layers.conv2d(
            data=data, in_dtype=dtype, channels=filter_list[0], kernel_size=(7, 7),
            strides=(2, 2), padding=(3, 3), data_layout=data_layout, kernel_layout=kernel_layout, name="conv0")

        ## Add multiply
        # body = layers.quantized_batchnorm(body, data_layout=data_layout, channels=filter_list[0], name="conv0_bn")

        body = relay.nn.relu(data=body)
        body = relay.nn.max_pool2d(data=body, pool_size=(3, 3), strides=(2, 2), padding=(1, 1), layout=data_layout)

    # quantize
    qconfig_conv0 = layers.get_qconfig("conv0_qconfig")
    # body = layers.quantize(body, output_scale=qconfig_conv0.output_scale, output_zero_point=qconfig_conv0.output_scale, out_dtype='int32')

    for i in range(num_stages):
        body = residual_unit(
            body, filter_list[i], filter_list[i+1], (1 if i == 0 else 2, 1 if i == 0 else 2),
            (filter_list[i] == filter_list[i+1]), name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck,
            data_layout=data_layout, kernel_layout=kernel_layout)
        for j in range(units[i]-1):
            body = residual_unit(
                body, filter_list[i+1], filter_list[i+1], (1, 1), True,
                name='stage%d_unit%d' % (i + 1, j + 2), bottle_neck=bottle_neck,
                data_layout=data_layout, kernel_layout=kernel_layout)

    relu1 = relay.nn.relu(data=body)
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = relay.nn.global_avg_pool2d(data=relu1, layout=data_layout)
    flat = relay.nn.batch_flatten(data=pool1)

    # dequantize
    qconfig_fc = layers.get_qconfig(name='fc1_qconfig')
    deq = layers.dequantize(flat, input_scale=qconfig_fc.input_scale, input_zero_point=qconfig_fc.input_scale)

    fc1 = layers.dense_add_bias(data=deq, units=num_classes, name='fc1')
    net = relay.nn.softmax(data=fc1)
    # return net
    return relay.Function(relay.analysis.free_vars(net), net)   # notice here is for create_wordload


def get_net(batch_size,
            num_classes,
            num_layers=50,
            image_shape=(3, 224, 224),
            data_layout="NCHW",
            kernel_layout="OIHW",
            dtype="float32",
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

    return qnn_resnet(units=units,
                  num_stages=num_stages,
                  filter_list=filter_list,
                  num_classes=num_classes,
                  data_shape=data_shape,
                  bottle_neck=bottle_neck,
                  data_layout=data_layout,
                  kernel_layout=kernel_layout,
                  dtype=dtype)



def get_workload(batch_size=1,
                 num_classes=1000,
                 num_layers=18,
                 image_shape=(3, 224, 224),
                 dtype="float32",
                 data_layout="NCHW",
                 kernel_layout="OIHW",
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
                  **kwargs)

    return create_workload(net, QuantizeInitializer())