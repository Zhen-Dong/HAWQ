import tvm
from tvm import relay

from . import layers
from .init import create_workload, QuantizeInitializer


def Conv(data, num_filter, input_channels, qconfig, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', data_layout="NCHW", kernel_layout="OIHW"):

    if input_channels != 3:
        data = layers.requantize(data,
                                input_scale=qconfig.from_scale,
                                input_zero_point=qconfig.from_zero_point,
                                output_scale=qconfig.input_scale,
                                output_zero_point=qconfig.input_zero_point,
                                out_dtype=qconfig.input_dtype)
    
    
    conv = layers.quantized_conv2d(data=data,
                                    name='%s%s_conv1' % (name, suffix),
                                    add_bias=True,
                                    input_channels=input_channels,
                                    output_channels=int(num_filter),
                                    kernel_dtype=qconfig.kernel_dtype,
                                    input_scale=qconfig.input_scale,
                                    kernel_scale=qconfig.kernel_scale,
                                    input_zero_point=qconfig.input_zero_point,
                                    kernel_zero_point=qconfig.kernel_zero_point,
                                    kernel_size=kernel,
                                    strides=stride, padding=pad,
                                    data_layout=data_layout, kernel_layout=kernel_layout)
    
    act = relay.nn.relu(data=conv)
    act = relay.annotation.stop_fusion(act)
    return act


def Pooling(data, kernel, stride, pad, pool_type, layout, name):
    if pool_type == 'max':
        return relay.nn.max_pool2d(data=data, pool_size=kernel, strides=stride, padding=pad, layout=layout)
    if pool_type == 'avg':
        return relay.nn.avg_pool2d(data=data, pool_size=kernel, strides=stride, padding=pad, layout=layout,
                                   count_include_pad=True)
    raise ValueError("Invalid pooling type: " + pool_type)


def Inception7A(data,
                num_1x1,
                num_3x3_red, num_3x3_1, num_3x3_2,
                num_5x5_red, num_5x5, input_channels,
                pool, proj, qconfigs,
                name,
                data_layout,
                kernel_layout):
    tower_1x1 = Conv(data, num_1x1, input_channels, qconfigs[0], name=('%s_conv' % name), data_layout=data_layout, kernel_layout=kernel_layout)
    tower_5x5 = Conv(data, num_5x5_red, input_channels, qconfigs[1], name=('%s_tower' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_5x5 = Conv(tower_5x5, num_5x5, num_5x5_red, qconfigs[2], kernel=(5, 5), pad=(2, 2), name=('%s_tower' % name),
                     suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3 = Conv(data, num_3x3_red, input_channels, qconfigs[3], name=('%s_tower_1' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3 = Conv(tower_3x3, num_3x3_1, num_3x3_red, qconfigs[4], kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name),
                     suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3 = Conv(tower_3x3, num_3x3_2, num_3x3_1, qconfigs[5], kernel=(3, 3), pad=(1, 1), name=('%s_tower_1' % name),
                     suffix='_conv_2', data_layout=data_layout, kernel_layout=kernel_layout)
    pooling = Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, layout=data_layout,
                      name=('%s_pool_%s_pool' % (pool, name)))

    cproj = Conv(pooling, proj, input_channels, qconfigs[6], name=('%s_tower_2' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    
    concat_input_scales = [qconfigs[0].input_scale * qconfigs[0].kernel_scale,
                          qconfigs[2].input_scale * qconfigs[2].kernel_scale,
                          qconfigs[5].input_scale * qconfigs[5].kernel_scale,
                          qconfigs[6].input_scale * qconfigs[6].kernel_scale]
    concat_input_zero_points = [0, 0, 0, 0]

    axis = 1
    if data_layout == 'NHWC':
        axis = 3

    concat = layers.quantized_concatenate((tower_1x1, tower_5x5, tower_3x3, cproj),
                                           concat_input_scales, 
                                           concat_input_zero_points,
                                           qconfigs[0].output_scale,
                                           qconfigs[0].output_zero_point,
                                           axis=axis)

    # concat = relay.concatenate((tower_1x1, tower_5x5, tower_3x3, cproj), axis=1)
    return concat


# First Downsample
def Inception7B(data,
                num_3x3,
                num_d3x3_red, num_d3x3_1, num_d3x3_2, input_channels,
                pool, qconfigs,
                name,
                data_layout,
                kernel_layout):
    tower_3x3 = Conv(data, num_3x3, input_channels, qconfigs[0], kernel=(3, 3), pad=(0, 0), stride=(2, 2),
                     name=('%s_conv' % name), data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d3x3 = Conv(data, num_d3x3_red, input_channels, qconfigs[1], name=('%s_tower' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_1, num_d3x3_red, qconfigs[2], kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                      name=('%s_tower' % name), suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d3x3 = Conv(tower_d3x3, num_d3x3_2, num_d3x3_1, qconfigs[3], kernel=(3, 3), pad=(0, 0), stride=(2, 2),
                      name=('%s_tower' % name), suffix='_conv_2', data_layout=data_layout, kernel_layout=kernel_layout)
    pooling = Pooling(data=data, kernel=(3, 3), stride=(2, 2), pad=(0, 0), pool_type="max", layout=data_layout,
                      name=('max_pool_%s_pool' % name))

    concat_input_scales = [qconfigs[0].input_scale * qconfigs[0].kernel_scale,
                          qconfigs[3].input_scale * qconfigs[3].kernel_scale,
                          qconfigs[0].from_scale]
    concat_input_zero_points = [0, 0, 0]
    axis = 1
    if data_layout == 'NHWC':
        axis = 3
    concat = layers.quantized_concatenate((tower_3x3, tower_d3x3, pooling),
                                           concat_input_scales, 
                                           concat_input_zero_points,
                                           qconfigs[0].output_scale,
                                           qconfigs[0].output_zero_point,
                                           axis=axis)    

    # concat = relay.concatenate((tower_3x3, tower_d3x3, pooling), axis=1)
    return concat


def Inception7C(data,
                num_1x1,
                num_d7_red, num_d7_1, num_d7_2,
                num_q7_red, num_q7_1, num_q7_2, num_q7_3, num_q7_4, input_channels,
                pool, proj, qconfigs,
                name,
                data_layout,
                kernel_layout):
    tower_1x1 = Conv(data, num_1x1, input_channels, qconfigs[0], kernel=(1, 1), name=('%s_conv' % name), data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d7 = Conv(data, num_d7_red, input_channels, qconfigs[1], name=('%s_tower' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d7 = Conv(tower_d7, num_d7_1, num_d7_red, qconfigs[2], kernel=(1, 7), pad=(0, 3),
                    name=('%s_tower' % name), suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d7 = Conv(tower_d7, num_d7_2, num_d7_1, qconfigs[3], kernel=(7, 1), pad=(3, 0),
                    name=('%s_tower' % name), suffix='_conv_2', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_q7 = Conv(data, num_q7_red, input_channels, qconfigs[4], name=('%s_tower_1' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_q7 = Conv(tower_q7, num_q7_1, num_q7_red, qconfigs[5], kernel=(7, 1), pad=(3, 0),
                    name=('%s_tower_1' % name), suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_q7 = Conv(tower_q7, num_q7_2, num_q7_1, qconfigs[6], kernel=(1, 7), pad=(0, 3),
                    name=('%s_tower_1' % name), suffix='_conv_2', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_q7 = Conv(tower_q7, num_q7_3, num_q7_2, qconfigs[7], kernel=(7, 1), pad=(3, 0),
                    name=('%s_tower_1' % name), suffix='_conv_3', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_q7 = Conv(tower_q7, num_q7_4, num_q7_3, qconfigs[8], kernel=(1, 7), pad=(0, 3),
                    name=('%s_tower_1' % name), suffix='_conv_4', data_layout=data_layout, kernel_layout=kernel_layout)
    pooling = Pooling(data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, layout=data_layout,
                      name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(pooling, proj, input_channels, qconfigs[9], kernel=(1, 1),
                 name=('%s_tower_2' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    
    # concat
    concat_input_scales = [qconfigs[0].input_scale * qconfigs[0].kernel_scale,
                          qconfigs[3].input_scale * qconfigs[3].kernel_scale,
                          qconfigs[8].input_scale * qconfigs[8].kernel_scale,
                          qconfigs[9].input_scale * qconfigs[9].kernel_scale]
    concat_input_zero_points = [0, 0, 0, 0]
    axis = 1
    if data_layout == 'NHWC':
        axis = 3
    concat = layers.quantized_concatenate((tower_1x1, tower_d7, tower_q7, cproj),
                                           concat_input_scales, 
                                           concat_input_zero_points,
                                           qconfigs[0].output_scale,
                                           qconfigs[0].output_zero_point,
                                           axis=axis)


    # concat = relay.concatenate((tower_1x1, tower_d7, tower_q7, cproj), axis=1)
    return concat

def Inception7D(data,
                num_3x3_red, num_3x3,
                num_d7_3x3_red, num_d7_1, num_d7_2, num_d7_3x3, input_channels,
                pool, qconfigs,
                name,
                data_layout,
                kernel_layout):
    tower_3x3 = Conv(data, num_3x3_red, input_channels, qconfigs[0], name=('%s_tower' % name),
                     suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3 = Conv(tower_3x3, num_3x3, num_3x3_red, qconfigs[1], kernel=(3, 3), pad=(0, 0), stride=(2, 2),
                     name=('%s_tower' % name), suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d7_3x3 = Conv(data, num_d7_3x3_red, input_channels, qconfigs[2], name=('%s_tower_1' % name),
                        suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d7_3x3 = Conv(tower_d7_3x3, num_d7_1, num_d7_3x3_red, qconfigs[3], kernel=(1, 7), pad=(0, 3),
                        name=('%s_tower_1' % name), suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d7_3x3 = Conv(tower_d7_3x3, num_d7_2, num_d7_1, qconfigs[4], kernel=(7, 1), pad=(3, 0),
                        name=('%s_tower_1' % name), suffix='_conv_2', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d7_3x3 = Conv(tower_d7_3x3, num_d7_3x3, num_d7_2, qconfigs[5], kernel=(3, 3), stride=(2, 2),
                        name=('%s_tower_1' % name), suffix='_conv_3', data_layout=data_layout, kernel_layout=kernel_layout)
    pooling = Pooling(data, kernel=(3, 3), stride=(2, 2), pool_type=pool, pad=(0, 0), layout=data_layout,
                      name=('%s_pool_%s_pool' % (pool, name)))
    # concat
    concat_input_scales = [qconfigs[1].input_scale * qconfigs[1].kernel_scale,
                          qconfigs[5].input_scale * qconfigs[5].kernel_scale,
                          qconfigs[0].from_scale]
    concat_input_zero_points = [0, 0, 0]
    axis = 1
    if data_layout == 'NHWC':
        axis = 3
    concat = layers.quantized_concatenate((tower_3x3, tower_d7_3x3, pooling),
                                           concat_input_scales, 
                                           concat_input_zero_points,
                                           qconfigs[0].output_scale,   # all output_scale store in qconfig0
                                           qconfigs[0].output_zero_point,
                                           axis=axis)    


    # concat = relay.concatenate((tower_3x3, tower_d7_3x3, pooling), axis=1)
    return concat

def Inception7E(data,
                num_1x1,
                num_d3_red, num_d3_1, num_d3_2,
                num_3x3_d3_red, num_3x3, num_3x3_d3_1, num_3x3_d3_2, input_channels,
                pool, proj, qconfigs,
                name,
                data_layout,
                kernel_layout):
    tower_1x1 = Conv(data, num_1x1, input_channels, qconfigs[0], kernel=(1, 1), name=('%s_conv' % name), data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d3 = Conv(data, num_d3_red, input_channels, qconfigs[1], name=('%s_tower' % name), suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d3_a = Conv(tower_d3, num_d3_1, num_d3_red, qconfigs[2], kernel=(1, 3), pad=(0, 1),
                      name=('%s_tower' % name), suffix='_mixed_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_d3_b = Conv(tower_d3, num_d3_2, num_d3_red, qconfigs[3], kernel=(3, 1), pad=(1, 0),
                      name=('%s_tower' % name), suffix='_mixed_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3_d3 = Conv(data, num_3x3_d3_red, input_channels, qconfigs[4], name=('%s_tower_1' % name),
                        suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3_d3 = Conv(tower_3x3_d3, num_3x3, num_3x3_d3_red, qconfigs[5], kernel=(3, 3), pad=(1, 1),
                        name=('%s_tower_1' % name), suffix='_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3_d3_a = Conv(tower_3x3_d3, num_3x3_d3_1, num_3x3, qconfigs[6], kernel=(1, 3), pad=(0, 1),
                          name=('%s_tower_1' % name), suffix='_mixed_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    tower_3x3_d3_b = Conv(tower_3x3_d3, num_3x3_d3_2, num_3x3, qconfigs[7], kernel=(3, 1), pad=(1, 0),
                          name=('%s_tower_1' % name), suffix='_mixed_conv_1', data_layout=data_layout, kernel_layout=kernel_layout)
    pooling = Pooling(data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, layout=data_layout,
                      name=('%s_pool_%s_pool' % (pool, name)))
    cproj = Conv(pooling, proj, input_channels, qconfigs[8], kernel=(1, 1), name=('%s_tower_2' % name),
                 suffix='_conv', data_layout=data_layout, kernel_layout=kernel_layout)
    # concat
    concat_input_scales = [qconfigs[0].input_scale * qconfigs[0].kernel_scale,
                          qconfigs[2].input_scale * qconfigs[2].kernel_scale,
                          qconfigs[3].input_scale * qconfigs[3].kernel_scale,
                          qconfigs[6].input_scale * qconfigs[6].kernel_scale,
                          qconfigs[7].input_scale * qconfigs[7].kernel_scale,
                          qconfigs[8].input_scale * qconfigs[8].kernel_scale]
    concat_input_zero_points = [0, 0, 0, 0, 0, 0]
    axis = 1
    if data_layout == 'NHWC':
        axis = 3
    concat = layers.quantized_concatenate((tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj),
                                           concat_input_scales, 
                                           concat_input_zero_points,
                                           qconfigs[0].output_scale,
                                           qconfigs[0].output_zero_point,
                                           axis=axis)

    # concat = relay.concatenate(
    #     (tower_1x1, tower_d3_a, tower_d3_b, tower_3x3_d3_a, tower_3x3_d3_b, cproj), axis=1)
    return concat



def get_net(batch_size,
            num_classes,
            image_shape,
            dtype,
            data_layout="NCHW",
            kernel_layout="OIHW",
            **kwargs):
    """Get network a Inception v3 network.

    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of claseses

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    net : relay.Function
        The dataflow.
    """
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
    data = relay.var("data",
                     shape=data_shape,
                     dtype=dtype)

    
    # if data_layout == 'NCHW':
    #     data_channel = image_shape[0]
    # elif data_layout == 'NHWC':
    #     data_channel = image_shape[2]


    # stage 1
    qconfig_conv = layers.get_qconfig("conv0_qconfig")
    conv = Conv(data, 32, 3, qconfig_conv, kernel=(3, 3), stride=(2, 2), name="conv", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_conv_1 = layers.get_qconfig("conv1_qconfig")
    conv_1 = Conv(conv, 32, 32, qconfig_conv_1, kernel=(3, 3), name="conv_1", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_conv_2 = layers.get_qconfig("conv2_qconfig")
    conv_2 = Conv(conv_1, 64, 32, qconfig_conv_2, kernel=(3, 3), pad=(1, 1), name="conv_2", data_layout=data_layout, kernel_layout=kernel_layout)
    pool = Pooling(data=conv_2, kernel=(3, 3), stride=(2, 2), pool_type="max", pad=(0, 0), layout=data_layout,
                   name="pool")
    # stage 2
    qconfig_conv_3 = layers.get_qconfig("conv3_qconfig")
    conv_3 = Conv(pool, 80, 64, qconfig_conv_3, kernel=(1, 1), name="conv_3", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_conv_4 = layers.get_qconfig("conv4_qconfig")
    conv_4 = Conv(conv_3, 192, 80, qconfig_conv_4, kernel=(3, 3), name="conv_4", data_layout=data_layout, kernel_layout=kernel_layout)
    pool1 = Pooling(data=conv_4, kernel=(3, 3), stride=(2, 2), pool_type="max", pad=(0, 0), layout=data_layout,
                    name="pool1")

    # stage 3
    qconfig_in3a = [layers.get_qconfig("in3a_conv%d" % (i)) for i in range(7)]
    in3a = Inception7A(pool1, 64,
                       64, 96, 96,
                       48, 64, 192,
                       "avg", 32, qconfig_in3a, "mixed", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in3b = [layers.get_qconfig("in3b_conv%d" % (i)) for i in range(7)]
    in3b = Inception7A(in3a, 64,
                       64, 96, 96,
                       48, 64, 256,
                       "avg", 64, qconfig_in3b, "mixed_1", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in3c = [layers.get_qconfig("in3c_conv%d" % (i)) for i in range(7)]
    in3c = Inception7A(in3b, 64,
                       64, 96, 96,
                       48, 64, 288,
                       "avg", 64, qconfig_in3c, "mixed_2", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in3d = [layers.get_qconfig("in3d_conv%d" % (i)) for i in range(4)]
    in3d = Inception7B(in3c, 384,
                       64, 96, 96, 288,
                       "max", qconfig_in3d, "mixed_3", data_layout=data_layout, kernel_layout=kernel_layout)
    # stage 4
    qconfig_in4a = [layers.get_qconfig("in4a_conv%d" % (i)) for i in range(10)]
    in4a = Inception7C(in3d, 192,
                       128, 128, 192,
                       128, 128, 128, 128, 192, 768,
                       "avg", 192, qconfig_in4a, "mixed_4", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in4b = [layers.get_qconfig("in4b_conv%d" % (i)) for i in range(10)]
    in4b = Inception7C(in4a, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192, 768,
                       "avg", 192, qconfig_in4b, "mixed_5", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in4c = [layers.get_qconfig("in4c_conv%d" % (i)) for i in range(10)]
    in4c = Inception7C(in4b, 192,
                       160, 160, 192,
                       160, 160, 160, 160, 192, 768,
                       "avg", 192, qconfig_in4c, "mixed_6", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in4d = [layers.get_qconfig("in4d_conv%d" % (i)) for i in range(10)]
    in4d = Inception7C(in4c, 192,
                       192, 192, 192,
                       192, 192, 192, 192, 192, 768,
                       "avg", 192, qconfig_in4d, "mixed_7", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in4e = [layers.get_qconfig("in4e_conv%d" % (i)) for i in range(6)]
    in4e = Inception7D(in4d, 192, 320,
                       192, 192, 192, 192, 768,
                       "max", qconfig_in4e, "mixed_8", data_layout=data_layout, kernel_layout=kernel_layout)
    # stage 5
    qconfig_in5a = [layers.get_qconfig("in5a_conv%d" % (i)) for i in range(9)]
    in5a = Inception7E(in4e, 320,
                       384, 384, 384,
                       448, 384, 384, 384, 1280,
                       "avg", 192, qconfig_in5a, "mixed_9", data_layout=data_layout, kernel_layout=kernel_layout)
    qconfig_in5b = [layers.get_qconfig("in5b_conv%d" % (i)) for i in range(9)]
    in5b = Inception7E(in5a, 320,
                       384, 384, 384,
                       448, 384, 384, 384, 2048,
                       "max", 192, qconfig_in5b, "mixed_10", data_layout=data_layout, kernel_layout=kernel_layout)

    # pool
    pool = Pooling(data=in5b, kernel=(8, 8), stride=(1, 1), pool_type="avg", pad=(0, 0), layout=data_layout,
                   name="global_pool")

    flatten = relay.nn.batch_flatten(pool)
    
    qconfig_fc = layers.get_qconfig(name='fc1_qconfig')
    
    req1 = layers.requantize(flatten, 
                            input_scale=qconfig_fc.from_scale, 
                            input_zero_point=qconfig_fc.from_zero_point,                             
                            output_scale=qconfig_fc.input_scale, 
                            output_zero_point=qconfig_fc.input_zero_point, 
                            out_dtype='int8')

    fc1 = layers.quantized_dense(data=req1, name='fc1', 
                                input_zero_point= qconfig_fc.input_zero_point,
                                kernel_zero_point=qconfig_fc.kernel_zero_point,
                                input_scale=qconfig_fc.input_scale,
                                kernel_scale=qconfig_fc.kernel_scale,
                                units=num_classes,
                                kernel_shape=(1000, 2048),
                                kernel_dtype='int8')
    
    deq = layers.dequantize(fc1, input_scale=qconfig_fc.input_scale * qconfig_fc.kernel_scale, input_zero_point=0.0)
    # fc1 = relay.nn.dense(flatten, relay.var("fc1_weight"), units=num_classes)
    # fc1 = relay.nn.bias_add(fc1, relay.var("fc2_bias"), axis=-1)
    inception_v3 = relay.nn.softmax(data=deq)
    args = relay.analysis.free_vars(inception_v3)
    return relay.Function(args, inception_v3)



def get_workload(batch_size=1, num_classes=1000,
                 image_shape=(3, 299, 299), dtype="int8",
                 data_layout="NCHW",
                 kernel_layout="OIHW",
                 **kwargs):
    """Get benchmark workload for InceptionV3

    Parameters
    ----------
    batch_size : int
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape

    dtype : str, optional
        The data type

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains an Inception V3 network.

    params : dict of str to NDArray
        The parameters.
    """
    net = get_net(batch_size, num_classes, image_shape, dtype,
                  data_layout=data_layout,
                  kernel_layout=kernel_layout,
                  **kwargs)
    return create_workload(net, QuantizeInitializer())
