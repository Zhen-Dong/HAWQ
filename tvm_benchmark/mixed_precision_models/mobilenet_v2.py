from tvm import relay
from init import create_workload

from . import layers

def conv_block(data, name, channels, kernel_size=(3, 3), strides=(1, 1),
               padding=(1, 1), epsilon=1e-5, layout='NCHW'):
    """Helper function to construct conv_bn-relu6"""
    # convolution + bn + relu6
    conv = layers.conv2d(
        data=data,
        channels=channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=conv_kernel_layout(layout),
        name=name+'_conv')
    bn = layers.batch_norm_infer(data=conv, epsilon=epsilon, name=name + '_bn')
    act = relay.nn.clip(bn, 0, 6)
    return act

def bottleneck_residual_block(data, name, out_channels, expansion, kernel_size=(3, 3),
                              strides=(1, 1), padding=(1, 1), epsilon=1e-5, layout='NCHW',
                              dtype='float32'):
    if layout == 'NHWC':
        in_channels = data.shape[3]
    else:
        in_channels = data.shape[1]     # assume NCHW

    hidden_dim = expansion * in_channels
    # pointwise convolution + bn + relu6
    conv1 = layers.conv2d(
        data=data,
        channels=hidden_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + '_conv1')

    bn1 = layers.batch_norm_infer(data=conv1, epsilon=epsilon, name=name + '_bn1')
    act1 = relay.nn.clip(bn1, 0, 6)
    # depthwise convolution + bn =+ relu6
    wshape = (hidden_dim, 1) + kernel_size
    weight = relay.var(name + "_weight", shape=wshape, dtype=dtype)
    conv2 = layers.conv2d(
        data=data,
        weight=weight,
        channels=hidden_dim,
        groups=hidden_dim,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout, True),
        name=name + '_depthwise_conv2')
    bn2 = layers.batch_norm_infer(data=conv2, epsilon=epsilon, name=name + '_bn2')
    act2 = relay.nn.clip(bn2, 0, 6)
    # linear pointwise convolution + bn
    conv3 = layers.conv2d(
        data=data,
        channels=out_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding=(0, 0),
        data_layout=layout,
        kernel_layout=layers.conv_kernel_layout(layout),
        name=name + '_conv3')
    bn3 = layers.batch_norm_infer(data=conv3, epsilon=epsilon, name=name + '_bn3')
    if in_channels == out_channels and strides == (1, 1):
        return relay.nn.add(bn3, data)
    else:
        return bn3

def mobile_net_v2(num_classes=1000, data_shape=(1, 3, 224, 224), dtype='float32',
                  layout='NCHW'):
    """Function to construct a MobileNetV2"""
    data = relay.var("data", shape=data_shape, dype=dtype)
    body = conv_block(data, 'conv_block_1', 32, strides=(2, 2), layout=layout)

    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1]
    ]

    block_num = 1
    for t, c, n, s in inverted_residual_setting:
        for i in range(n):
            stride = s if i == 0 else 1
            body = bottleneck_residual_block(data, 'bottleneck_residual_block_%d' % block_num,
                                             c, t, strides=(stride, stride),
                                             layout=layout, dtype=dtype)
            block_num += 1
    body = conv_block(data, 'conv_block_2', 1280, kernel_size=(1, 1),
                      padding=(0, 0), layout=layout)
    pool = relay.nn.global_avg_pool2d(data=body, layout=layout)
    flatten = relay.nn.batch_flatten(data=pool)
    fc = layers.dense_add_bias(data=flat, units=num_classes, name='fc')
    softmax = relay.nn.softmax(data=fc)
    return relay.Function(relay.analysis.free_vars(softmax), softmax)

def get_workload(batch_size=1, num_classes=1000, image_shape=(3, 224, 224),
                 dtype='float32', layout='NCHW'):
    """Get benchmark workload for mobilenet

    Parameters
    ----------
    batch_size : int, optional
        The batch size used in the model

    num_classes : int, optional
        Number of classes

    image_shape : tuple, optional
        The input image shape, cooperate with layout

    dtype : str, optional
        The data type

    layout : str, optional
        The data layout of image_shape and the operators
        cooperate with image_shape

    Returns
    -------
    mod : tvm.IRModule
        The relay module that contains a MobileNet network.

    params : dict of str to NDArray
        The parameters.
    """
    data_shape = tuple([batch_size] + list(image_shape))
    net = mobile_net_v2(num_classes=num_classes, data_shape=data_shape,
                     dtype=dtype, layout=layout)
    return create_workload(net)
