import torch
import numpy as np
import sys
import argparse

import os
from os.path import join

from mixed_precision_models.layers import QConfig, QuantizeContext
import scipy.signal
from numpy.lib.stride_tricks import as_strided
import tvm
from topi.nn.util import get_pad_tuple

###############################################################################
# Save weights
# -----------------

# Return an array where each int32 element stores eight 4-bit values from a_int32.
# Assumes each element in a_int32 can fit in 4-bit precision.
def pack_int32_to_int4(a_int32):
    I, J, K, L = a_int32.shape
    a_int4 = np.zeros(shape=(I, J, K, L // 8), dtype=np.int32)
    for i in range(I):
        for j in range(J):
            for k in range(K):
                for l in range(L // 8):
                    for m in range(min(8, L-l*8)):
                        a_int4[i, j, k, l] = a_int4[i, j, k, l] | ((a_int32[i, j, k, l * 8 + m] & 0xf) << ((7 - m) * 4))
    return a_int4

def unpack_int4_to_int32(a_int4):
    I, J, K, L = a_int4.shape
    a_int32 = np.zeros(shape=(I, J, K, L * 8), dtype=np.int32)
    for i in range(I):
            for j in range(J):
                for k in range(K):
                    for l in range(L):
                        for m in range(8):
                            a_int32[i, j, k, l * 8 + m] = (a_int4[i, j, k, l] >> ((7 - m) * 4)) & 0xf

    return a_int32

def conv2d_nhwc_python(a_np, w_np, w_layout, stride, padding):
    """Convolution operator in NHWC layout.

    Parameters
    ----------
    a_np : numpy.ndarray
        4-D with shape [batch, in_height, in_width, in_channel]

    w_np : numpy.ndarray
        4-D with shape [num_filter, filter_height, filter_width, in_channel]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or str or a list/tuple of 2 or 4 ints
        Padding size, or ['VALID', 'SAME'], or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 2 ints

    Returns
    -------
    b_np : np.ndarray
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    batch, in_height, in_width, in_channel = a_np.shape
    if w_layout == "OHWI":
        num_filter, kernel_h, kernel_w, _ = w_np.shape
    elif w_layout == "HWOI":
        kernel_h, kernel_w, num_filter, _ = w_np.shape

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (kernel_h, kernel_w))
    pad_h = pad_top + pad_bottom
    pad_w = pad_left + pad_right

    # compute the output shape
    out_channel = num_filter
    out_height = (in_height - kernel_h + pad_h) // stride_h + 1
    out_width = (in_width - kernel_w + pad_w) // stride_w + 1
    # change the layout from NHWC to NCHW
    at = a_np.transpose((0, 3, 1, 2))

    if w_layout == "OHWI":
        wt = w_np.transpose((0, 3, 1, 2))
    elif w_layout == "HWOI":
        wt = w_np.transpose((2, 3, 0, 1))

    bt = np.zeros((batch, out_channel, out_height, out_width))
    # computation
    for n in range(batch):
        for f in range(out_channel):
            for c in range(in_channel):
                if pad_h > 0 or pad_w > 0:
                    apad = np.zeros((in_height + pad_h, in_width + pad_w))
                    apad[pad_top:pad_top + in_height, pad_left:pad_left + in_width] = at[n, c]
                else:
                    apad = at[n, c]
                out = scipy.signal.convolve2d(
                    apad, np.rot90(np.rot90(wt[f, c])), mode='valid')
                bt[n, f] += out[::stride_h, ::stride_w]
    return bt.transpose((0, 2, 3, 1))


def save_weights(save_path, kernel_dtype, num_stages, units):
    params = {}

    int8_ops_list = ['module.quant_init_convbn.weight_integer', 'module.quant_output.weight_integer']

    for (key, tensor) in weight_integer.items():
        print(key)
        tensor_np = tensor.cpu().numpy().astype('int32')
        print(tensor_np.shape)
        # If it is convolutio weight, transpose it
        if len(tensor_np.shape) == 4:
            tensor_np = np.transpose(tensor_np, (2, 3, 0, 1))

        if key in int8_ops_list:
            tensor_np = tensor_np.astype('int8')

        if kernel_dtype == 'int4' and key not in int8_ops_list:
            tensor_np = pack_int32_to_int4(tensor_np)
        else:
            tensor_np = tensor_np.astype('int8')

        params[key] = tensor_np


    renamed_params = {}
    renamed_params['conv0_weight'] = params['module.quant_init_convbn.weight_integer']

    for i in range(num_stages):
        for j in range(units[i]):
            for k in range(3):
                old_name = "module.stage%d.unit%d.quant_convbn%d.weight_integer" % (i + 1, j + 1, k + 1)
                new_name = "stage%d_unit%d_qconv%d_weight" % (i + 1, j + 1, k + 1)
                assert old_name in params.keys(), "%s is not in the params" % old_name
                renamed_params[new_name] = params[old_name]

            if j == 0:
                old_name = "module.stage%d.unit%d.quant_identity_convbn.weight_integer" % (i + 1, j + 1)
                new_name = "stage%d_unit%d_qsc_weight" % (i + 1, j + 1)
                assert old_name in params.keys(), "%s is not in the params" % old_name
                renamed_params[new_name] = params[old_name]

    renamed_params['fc_weight'] = params['module.quant_output.weight_integer']
    np.save(os.path.join(save_path, "weights.npy"), renamed_params)

def load_qconfig_from_bit_config(num_stages, units, bit_config, bottleneck):

    def get_dtype (bit_width):
        assert bit_width == 4 or bit_width == 8, "Bit width %d not supported" % bit_width
        if bit_width == 4:
            data_dtype = "uint4"
            kernel_dtype = "int4"
        elif bit_width == 8:
            data_dtype = "int8"
            kernel_dtype = "int8"

        return data_dtype, kernel_dtype

    def load_conv_config(stage, unit, conv):
        bit_width = bit_config["stage%d.unit%d.quant_convbn%d" % (stage, unit, conv)]
        data_dtype, kernel_dtype = get_dtype(bit_width)
        QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig%d" % (stage, unit, conv)] = \
                QConfig(input_dtype=data_dtype, kernel_dtype=kernel_dtype)

    def load_sc_config(stage, unit):
        bit_width = bit_config["stage%d.unit%d.quant_identity_convbn" % (stage, unit)]
        data_dtype, kernel_dtype = get_dtype(bit_width)
        QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig_sc" % (stage, unit)] = \
            QConfig(input_dtype=data_dtype, kernel_dtype=kernel_dtype)

    QuantizeContext.qconfig_dict["conv0_qconfig"] = \
        QConfig(from_scale=1.0, input_dtype='int8', kernel_dtype='int8')

    for i in range(num_stages):
        for j in range(units[i]):

            if bottleneck:
                conv_num = 3
            else:
                conv_num = 2

            for k in range(conv_num):
                load_conv_config(i+1,j+1,k+1)

            if j == 0 and not (i == 0 and not bottleneck):
                load_sc_config(i+1, j+1)

    QuantizeContext.qconfig_dict["fc_qconfig"] = \
        QConfig(input_dtype='int8', kernel_dtype='int8')


###############################################################################
# Save scales
# -----------------

def load_qconfig(data_dtype, kernel_dtype, num_stages, units, model_load=False, scaling_factors=None, file_name=None):

    if not model_load:
        model = torch.load(file_name)
        scaling_factors = {**model['convbn_scaling_factor'], **model['fc_scaling_factor'], **model['act_scaling_factor']}

    params = {}
    for (key, tensor) in scaling_factors.items():
        tensor_np = tensor.cpu().numpy().reshape((-1))

        if "act_scaling_factor" in key:
            if np.ndim(tensor_np) == 1:
                tensor_np = tensor_np[0]

        params[key] = tensor_np

    def load_conv_config(stage, unit, conv):

        cur_hawq_conv = "module.stage%d.unit%d.quant_convbn%d.convbn_scaling_factor" % (stage, unit, conv)
        assert cur_hawq_conv in params.keys(), cur_hawq_conv + " does not exist"
        kernel_scale = params[cur_hawq_conv]

        if conv == 1:
            last_conv_unit, last_conv_stage = unit - 1, stage

            if last_conv_unit == 0:
                last_conv_stage = stage - 1
                last_conv_unit = units[last_conv_stage-1]

            if stage == 1 and unit == 1:
                last_conv = "conv0_qconfig"
            else:
                last_conv = "stage%d_unit%d_qconfig_add" % (last_conv_stage, last_conv_unit)

            assert last_conv in QuantizeContext.qconfig_dict.keys(), last_conv + " doesn't exist"
            from_scale = QuantizeContext.qconfig_dict[last_conv].output_scale

            # if stage == 1 and unit == 1:
            #     last_hawq_conv = "module.init_block.conv.3.act_scaling_factor"
            # else:
            #     last_hawq_conv = "module.stage%d.unit%d.quant_act.act_scaling_factor" % (last_conv_stage, last_conv_unit)

            last_hawq_conv = "module.stage%d.unit%d.quant_act.act_scaling_factor" % (stage, unit)

            assert last_hawq_conv in params.keys(), last_hawq_conv + " doesn't exist"

            input_scale = params[last_hawq_conv]
            output_scale = kernel_scale * input_scale
            QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig%d" % (stage, unit, conv)] = \
                QConfig(from_scale=from_scale, input_dtype=data_dtype, input_scale=input_scale, kernel_dtype=kernel_dtype, kernel_scale=kernel_scale, output_scale=output_scale)

        else:
            last_conv = "stage%d_unit%d_qconfig%d" % (stage, unit, conv-1)
            assert last_conv in QuantizeContext.qconfig_dict.keys(), last_conv + " doesn't exist"

            from_scale = QuantizeContext.qconfig_dict[last_conv].output_scale

            last_hawq_conv = "module.stage%d.unit%d.quant_act%d.act_scaling_factor" % (stage, unit, conv-1)
            assert last_hawq_conv in params.keys(), last_hawq_conv + " doesn't exist"
            input_scale = params[last_hawq_conv]

            output_scale = kernel_scale * input_scale

            QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig%d" % (stage, unit, conv)] = \
                QConfig(from_scale=from_scale, input_dtype=data_dtype, input_scale=input_scale, kernel_dtype=kernel_dtype, kernel_scale=kernel_scale, output_scale=output_scale)

    def load_sc_config(stage, unit):
        kernel_scale = params["module.stage%d.unit%d.quant_identity_convbn.convbn_scaling_factor" % (stage, unit)]
        input_scale = QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig1" % (stage, unit)].input_scale
        output_scale = kernel_scale * input_scale
        QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig_sc" % (stage, unit)] = \
            QConfig(input_dtype=data_dtype, input_scale=input_scale, kernel_dtype=kernel_dtype, kernel_scale=kernel_scale, output_scale=output_scale)

    def load_add_config(stage, unit, dim_match):
        lhs_output_scale = QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig3" % (stage, unit)].output_scale
        if dim_match:
            rhs_output_scale = QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig1" % (stage, unit)].from_scale
        else:
            rhs_output_scale = QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig_sc" % (stage, unit)].output_scale

        # output_scale = np.minimum(lhs_output_scale, rhs_output_scale)
        output_scale = params["module.stage%d.unit%d.quant_act_int32.act_scaling_factor" % (stage, unit)]
        QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig_add" % (stage, unit)] = QConfig(output_scale=output_scale)

    conv0_input_scale = params["module.quant_input.act_scaling_factor"]
    conv0_kernel_scale = params["module.quant_init_convbn.convbn_scaling_factor"]
    # conv0_output_scale = conv0_input_scale * conv0_kernel_scale
    conv0_output_scale = params["module.quant_act_int32.act_scaling_factor"]
    QuantizeContext.qconfig_dict["conv0_qconfig"] = \
        QConfig(from_scale=1.0, input_dtype='int8', input_scale=conv0_input_scale, kernel_dtype='int8', kernel_scale=conv0_kernel_scale, output_scale=conv0_output_scale)

    for i in range(num_stages):
        for j in range(units[i]):
            for k in range(3):
                load_conv_config(i+1,j+1,k+1)

            if j == 0:
                load_sc_config(i+1, j+1)
                load_add_config(i+1, j+1, False)
            else:
                load_add_config(i+1, j+1, True)

    fc_from_scale = QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig_add" % (num_stages, units[num_stages-1])].output_scale
    fc_input_scale = params["module.quant_act_output.act_scaling_factor"]
    fc_kernel_scale = params["module.quant_output.fc_scaling_factor"]
    fc_output_scale = (fc_input_scale * fc_kernel_scale)
    QuantizeContext.qconfig_dict["fc_qconfig"] = \
        QConfig(from_scale=fc_from_scale, input_dtype='int8', input_scale=fc_input_scale, kernel_dtype='int8', kernel_scale=fc_kernel_scale, output_scale=fc_output_scale)

# for (key, qconfig) in QuantizeContext.qconfig_dict.items():
#     print(key + ", %f, %f, %s, %f, %s, %f" % (qconfig.from_scale, qconfig.input_scale, qconfig.input_dtype, qconfig.kernel_scale, qconfig.kernel_dtype, qconfig.output_scale))

###############################################################################
# Save input_image
# -----------------
def save_input(model_dir, save_path):
    input_image = torch.load(os.path.join(model_dir, "input_image.pth.tar")).cpu().numpy()
    print(input_image.shape)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.transpose(input_image, (0, 2, 3, 1))
    np.save(os.path.join(save_path, "input_image_batch_1.npy"), input_image)

    #input_quantized_image = torch.load(os.path.join(model_dir, "quantized_input.pth.tar")).cpu().numpy()
    #input_quantized_image = np.transpose(input_quantized_image, (0, 2, 3, 1)) / QuantizeContext.qconfig_dict['conv0_qconfig'].input_scale
    #np.save(os.path.join(save_path, "quantized_input_image_batch_1.npy"), input_quantized_image)

###############################################################################
# Save scaled bias
# -----------------
def save_bias(save_path, num_stages, units):
    params = {}
    for (key, tensor) in scaled_bias.items():
        params[key] = tensor.cpu().numpy().reshape(1, 1, 1, -1)
        print(key)
        # print(tensor)

    renamed_params = {}
    renamed_params['conv0_bias'] = (params['module.quant_init_convbn.bias_integer']).astype("int32")

    for i in range(num_stages):
        for j in range(units[i]):
            for k in range(3):
                old_name = "module.stage%d.unit%d.quant_convbn%d.bias_integer" % (i + 1, j + 1, k + 1)
                new_name = "stage%d_unit%d_qconv%d_bias" % (i + 1, j + 1, k + 1)

                assert old_name in params.keys(), "%s is not in the params" % old_name

                renamed_params[new_name] = (params[old_name]).astype("int32")

            if j == 0:
                old_name = "module.stage%d.unit%d.quant_identity_convbn.bias_integer" % (i + 1, j + 1)
                new_name = "stage%d_unit%d_qsc_bias" % (i + 1, j + 1)

                assert old_name in params.keys(), "%s is not in the params" % old_name

                renamed_params[new_name] = (params[old_name]).astype("int32")

    renamed_params['fc_bias'] = (params['module.quant_output.bias_integer']).astype("int32")
    renamed_params['fc_bias'] = renamed_params['fc_bias'][0, 0, 0, :]
    for (key, tensor) in renamed_params.items():
        print(key)
        # print(tensor)

    np.save(os.path.join(save_path, "bias.npy"), renamed_params)


###############################################################################
# Save unit input
# -----------------
def save_unit_input(model_dir, save_path, num_stages, units):
    dir_path = os.path.join(save_path, "pytorch_result")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for i in range(num_stages):
        for j in range(units[i]):
            print("Stage %d Unit %d" % (i + 1, j + 1))
            golden_result = feature_map["module.stage%d.unit%d.block_input_featuremap" % (i+1, j+1)].cpu().numpy()[0].transpose((1, 2, 0))

            np.save(os.path.join(dir_path, "stage%d_unit%d_input_int4" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.block_before_act_input_featuremap" % (i+1, j+1)].cpu().numpy()[0].transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_input_int32" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.block_before_act_output_featuremap" % (i+1, j+1)].cpu().numpy().transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_output_int32" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.block_fp_output_featuremap" % (i+1, j+1)].cpu().numpy().transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_output_float32" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.convbnrelu1_before_act_output_featuremap" % (i+1, j+1)].cpu().numpy()[0].transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_conv1_output_int32" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.convbnrelu1_output_featuremap" % (i+1, j+1)].cpu().numpy()[0].transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_conv1_output_int4" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.convbnrelu2_output_featuremap" % (i+1, j+1)].cpu().numpy()[0].transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_conv2_output_int4" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.convbn3_output_featuremap" % (i+1, j+1)].cpu().numpy()[0].transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_conv3_output_int32" % (i+1, j+1)), golden_result)

            golden_result = feature_map["module.stage%d.unit%d.identity_featuremap" % (i+1, j+1)].cpu().numpy()[0].transpose((1, 2, 0))
            np.save(os.path.join(dir_path, "stage%d_unit%d_identity_int32" % (i+1, j+1)), golden_result)

    print("avg_pooling")
    golden_result = torch.load(os.path.join(model_dir, "average_pooling.pth.tar")).cpu().numpy().transpose((1, 2, 0))
    np.save(os.path.join(dir_path, "avg_pool_int32"), golden_result)

    print(golden_result.shape)

    print("fc")
    golden_result = torch.load(os.path.join(model_dir, "final_result.pth.tar")).cpu().numpy()
    np.save(os.path.join(dir_path, "fc_output_int32"), golden_result)

    # print("fc_float32")
    # golden_result = torch.load(os.path.join(model_dir, "final_result_fp32.pth.tar")).cpu().numpy()
    # np.save(os.path.join(dir_path, "fc_output_float32"), golden_result)

    # print(golden_result.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HWAQ-V3 utils',
                                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-dir', required=True,
                        help='Model data directory')

    parser.add_argument('--save-dir', required=False,
                        help='Saved parameters directory')

    parser.add_argument('--cifar10', action='store_true',
                        help='Model is cifar10')

    parser.add_argument('--with-featuremap', action='store_true',
                        help='Transform the featuremap')

    parser.add_argument('--onlyfeaturemap', action='store_true',
                        help='Only transform the featuremap')

    parser.add_argument('--dtype', default='int8',
                        help='Only support uniform data type here (int8, int4)')

    args = parser.parse_args()

    if args.save_dir is None:
        save_path = args.model_dir
    else:
        save_path = args.save_dir

    if args.dtype == 'int4':
        data_dtype = 'uint4'
        kernel_dtype = 'int4'
    elif args.dtype =='int8':
        data_dtype = 'int8'
        kernel_dtype = 'int8'
    else:
        sys.exit("dtype not supported")

    model_dir = args.model_dir
    file_name = os.path.join(model_dir, "quantized_checkpoint.pth.tar")
    input_file_name = "input.pth.tar"

    if save_path != model_dir:
        copy_command = "cp %s %s" % (file_name, save_path)
        os.system(copy_command)

    if args.cifar10:
        num_stages = 3
        units = [3, 4, 6]
    else:
        num_stages = 4
        units = [3, 4, 6, 3]

    model = torch.load(file_name)

    print(model.keys())

    weight_integer = model['weight_integer']
    scaling_factors = {**model['convbn_scaling_factor'], **model['fc_scaling_factor'], **model['act_scaling_factor']}
    scaled_bias = {**model['bias_integer']}


    if args.onlyfeaturemap or args.with_featuremap:
        save_input(model_dir, save_path)
        featuremap_name = os.path.join(model_dir, "featuremaps.pth.tar")
        feature_map = torch.load(featuremap_name)['featuremap']

        for key in feature_map.keys():
            print(key)

        save_unit_input(model_dir, save_path, num_stages, units)

    if not args.onlyfeaturemap:
        save_weights(save_path, kernel_dtype, num_stages, units)
        load_qconfig(data_dtype, kernel_dtype, num_stages, units, model_load=True, scaling_factors=scaling_factors)
        save_bias(save_path, num_stages, units)
