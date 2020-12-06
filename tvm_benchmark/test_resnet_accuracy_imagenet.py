import torch
import tvm
from tvm import autotvm
from tvm import relay

import sys
import os
import time
import argparse

import mxnet as mx
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import hawq_utils_resnet50
sys.path.append('..')
import mixed_precision_models.quantized_resnet_v1 as quantized_resnet_v1
from mixed_precision_models.layers import QConfig, QuantizeContext


def get_params(params_dir, model_type):
    if model_type == 'int4':
        data_dtype = 'uint4'
        kernel_dtype = 'int4'
    elif model_type == 'int8':
        kernel_dtype = data_dtype = 'int8'
    else:
        print("Model type not supported")
        sys.exit(1)

    num_stages = 4
    units = [3, 4, 6, 3]

    weights = np.load(os.path.join(params_dir, "weights.npy"), allow_pickle=True)[()]
    bias = np.load(os.path.join(params_dir, "bias.npy"), allow_pickle=True)[()]
    hawq_utils_resnet50.load_qconfig(data_dtype, kernel_dtype, num_stages=num_stages, units=units, file_name=os.path.join(params_dir, "quantized_checkpoint.pth.tar"))

    params = {**weights, **bias}
    # params = {**weights}
    return params


def get_model(num_layers, model_type, batch_size):
    image_shape = (3, 224, 224)
    input_dtype = 'int8'
    data_layout = "NHWC"
    kernel_layout = "HWOI"

    func, _ = quantized_resnet_v1.get_workload(batch_size=batch_size,
                                                    image_shape=image_shape,
                                                    num_layers=num_layers,
                                                    dtype=input_dtype,
                                                    data_layout=data_layout,
                                                    kernel_layout=kernel_layout,
                                                    with_bn=False,
                                                    with_softmax=False)

    return func


def quantize_image(image, rounding="TONEAREST"):
    image = np.transpose(image, (0, 2, 3, 1))
    image = image / QuantizeContext.qconfig_dict["conv0_qconfig"].input_scale
    image = np.clip(image, -128, 127)

    if rounding == "TONEAREST":
        image = np.round(image)
    elif rounding == "TRUNCATE":
        image = np.trunc(image)
    else:
        print("Unsupported rounding method")

    image = image.astype("int8")
    return image


def validate(val_dir, params_dir, batch_size, num_layers, model_type, log_filename=None, log_interval=10):
    TARGET_NAME = 'cuda'
    CTX = tvm.context(TARGET_NAME, 0)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # Get weight, bias and scaling factors params
    params = get_params(params_dir, model_type)

    # Get TVM quantized model
    func = get_model(num_layers, model_type, batch_size)

    # Build TVM runtime graph
    with autotvm.apply_history_best(log_filename):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(func, target=TARGET_NAME, params=params)

    module = tvm.contrib.graph_runtime.create(graph, lib, ctx=CTX)
    module.set_input(**params)

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()

    for i, (images, target) in enumerate(val_loader):

            images = quantize_image(images.cpu().numpy())
            # print (images.shape)
            # print (images)

            # print(target.cpu().numpy())

            # compute output
            module.set_input('data', images)
            module.run()

            tvm_output = module.get_output(0)

            acc_top1.update([mx.nd.array(target.cpu().numpy())], [mx.nd.array(tvm_output.asnumpy())])
            acc_top5.update([mx.nd.array(target.cpu().numpy())], [mx.nd.array(tvm_output.asnumpy())])

            if not (i + 1) % log_interval:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                nsamples = (i + 1) * batch_size
                print("[%d samples] validation: acc-top1=%f acc-top5=%f" % (nsamples, top1, top5))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Resnet50 imagenet accuracy test',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model-dir', required=True,
                        help='Model data directory')

    parser.add_argument('--val-dir', required=True, default=None,
                        help='Validation dataset directory')

    args = parser.parse_args()

    batch_size = 8

    val_dir = args.val_dir
    params_dir = args.model_dir

    num_layers = 50
    model_type = "int8"

    log_filename = "./logs/resnet%d_%s_%s_batch_%d.log" % (num_layers, "NHWC", model_type, batch_size)
    if not os.path.exists(log_filename):
        log_filename = None
    else:
        print("Apply tuning log " + log_filename)


    validate(val_dir, params_dir, batch_size, num_layers, model_type, log_filename=log_filename)
