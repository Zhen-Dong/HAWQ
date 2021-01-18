import torch
import tvm
from tvm import autotvm
from tvm import relay
from tvm.contrib import download
from tvm.contrib.debugger import debug_runtime

from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import argparse
import os
from os.path import join, isfile
import sys
import json, requests
from io import BytesIO
import re

import mixed_precision_models.quantized_resnet_v1 as quantized_resnet_v1
from mixed_precision_models.layers import QConfig, QuantizeContext

import hawq_utils_resnet50

import logging
logging.basicConfig(level=logging.CRITICAL)

parser = argparse.ArgumentParser(description='Resnet accuracy test',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model-dir', required=True,
                    help='Model data directory')

parser.add_argument('--debug-unit', default=None,
                    help='Debug specific unit input, compare the unit input to the pytorch result (stage1_unit1, stage1_unit2 ...)')

parser.add_argument('--rounding', default='TONEAREST',
                    help='Round scheme (TONEAREST, TRUNCATE)')

parser.add_argument('--num-classes', type=int, default=1000,
                    help='Total number of classes')

args = parser.parse_args()

###############################################################################
# Set target device
# -----------------

TARGET_NAME = 'cuda'
CTX = tvm.context(TARGET_NAME, 0)

###############################################################################
# Load params
# -----------------

if args.num_classes == 10: # Cifar 10
    num_stages = 3
    units = [3, 4, 6]
    print("Use Cifar 10")
else:
    num_stages = 4
    units = [3, 4, 6, 3]

weights = np.load(os.path.join(args.model_dir, "weights.npy"), allow_pickle=True)[()]
bias = np.load(os.path.join(args.model_dir, "bias.npy"), allow_pickle=True)[()]
hawq_utils_resnet50.load_qconfig("uint4", "int4", num_stages, units, file_name=os.path.join(args.model_dir, "quantized_checkpoint.pth.tar"))

input_image = np.load(os.path.join(args.model_dir, "input_image_batch_1.npy"))
input_image = input_image / QuantizeContext.qconfig_dict["conv0_qconfig"].input_scale
input_image = np.clip(input_image, -128, 127)

if args.rounding == "TONEAREST":
    input_image = np.round(input_image)
elif args.rounding == "TRUNCATE":
    input_image = np.trunc(input_image)

input_image = input_image.astype("int8")
params = {**weights, **bias}

###############################################################################
# Load model
# -----------------
batch_size = 8
shape = list(input_image.shape)
image_shape = (shape[3], shape[1], shape[2])
input_dtype = 'int8'
model_type = "int4"
num_layers = 50
data_layout = "NHWC"
kernel_layout = "HWOI"

func, _ = quantized_resnet_v1.get_workload(batch_size=batch_size,
                                                image_shape=image_shape,
                                                num_classes=args.num_classes,
                                                num_layers=num_layers,
                                                dtype=input_dtype,
                                                data_layout=data_layout,
                                                kernel_layout=kernel_layout,
                                                with_bn=False,
                                                debug_unit=args.debug_unit,
                                                rounding=args.rounding)


# Download ImageNet categories
categ_url = "https://github.com/uwsaml/web-data/raw/main/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

image = input_image
input_data = np.repeat(image, batch_size, axis=0)

###############################################################################
# Run the model
# -----------------
log_filename = "/home/zach_zheng/hawq_tvm/mixed_precision_models/tuning_logs/resnet%d_%s_%s_batch_%d.log" % (num_layers, data_layout, model_type, batch_size)

if not os.path.exists(log_filename):
    log_filename = None
else:
    print("Apply tuning log " + log_filename)

with autotvm.apply_history_best(log_filename):
    with relay.build_config(opt_level=3):

        graph, lib, params = relay.build(func, target=TARGET_NAME, params=params)

        if args.debug_unit is not None:
            m = tvm.contrib.graph_runtime.create(graph, lib, CTX)

            # Set the network parameters and inputs
            m.set_input(**params)
            m.set_input('data', input_data)
            m.run()

            np.set_printoptions(threshold=sys.maxsize)

            out = m.get_output(0).asnumpy()

            if not os.path.exists(os.path.join(args.model_dir, "tvm_result")):
                os.mkdir(os.path.join(args.model_dir, "tvm_result"))

            unit_str_regex = re.search('stage(\d)_unit(\d)', args.debug_unit)
            if unit_str_regex is not None:
                unit_str = unit_str_regex.group(0)
            else:
                unit_str = ""

            if args.debug_unit == "fc_input":
                actual_result = out
                np.save(os.path.join(args.model_dir, "tvm_result/fc_input_int8.npy"), actual_result[0])
                golden_result = np.load(os.path.join(args.model_dir, "pytorch_result/fc_input_int8.npy")).astype("int8")
            elif args.debug_unit == "fc_output":
                golden_result = np.load(os.path.join(args.model_dir, "pytorch_result/fc_output_int32.npy"))
                actual_result = out
                np.save(os.path.join(args.model_dir, "tvm_result/fc_output_int32.npy"), actual_result[0])
                # golden_result = np.load(os.path.join(args.model_dir, "pytorch_result/fc_output_float32.npy"))#.astype("int32")
            elif args.debug_unit == "avg_pool":
                actual_result = out
                np.save(os.path.join(args.model_dir, "tvm_result/avg_pool_int32.npy"), actual_result[0])
                golden_result = np.load(os.path.join(args.model_dir, "pytorch_result/avg_pool_int32.npy")).astype("int32")
            elif args.debug_unit == "softmax":
                actual_result = out
                np.save(os.path.join(args.model_dir, "tvm_result/avg_pool_int32.npy"), actual_result[0])
                golden_result = np.load(os.path.join(args.model_dir, "pytorch_result/avg_pool_int32.npy")).astype("int32")
            elif args.debug_unit == unit_str + "_output":
                actual_result = out * QuantizeContext.qconfig_dict["%s_qconfig_add" % unit_str].output_scale
                # actual_result = out
                np.save(os.path.join(args.model_dir, "tvm_result/%s_output_int32.npy" % unit_str), actual_result[0])
                golden_result = np.load(os.path.join(args.model_dir, "pytorch_result/%s_output_float32.npy" % unit_str))
            elif args.debug_unit == unit_str + "_input":
                actual_result = hawq_utils_resnet50.unpack_int4_to_int32(out)
                np.save(os.path.join(args.model_dir, "tvm_result/%s_input_int4.npy" % unit_str), actual_result[0])
                golden_result = np.load(os.path.join(args.model_dir, "pytorch_result/%s_input_int4.npy" % unit_str)).astype("int32")
            else:
                print("Error: Unsupported debug unit.")

            print("Above is Pytorch result, under is TVM result")
            tvm.testing.assert_allclose(golden_result, actual_result[0])

            print(args.debug_unit + " is 100% matched !")
        else:
            module = tvm.contrib.graph_runtime.create(graph, lib, ctx=CTX)
            module.set_input(**params)
            module.set_input('data', input_data)

            module.run()

            tvm_output = module.get_output(0)

            print(tvm_output.shape)

            for b in range(batch_size):
                top_categories = np.argsort(tvm_output.asnumpy()[b])
                # Report top-5 classification results
                print("\n prediction for sample {}".format(b))
                print("\t#1:", synset[top_categories[-1]])
                print("\t#2:", synset[top_categories[-2]])
                print("\t#3:", synset[top_categories[-3]])
                print("\t#4:", synset[top_categories[-4]])
                print("\t#5:", synset[top_categories[-5]])
