import tvm

from tvm import relay
from tvm import autotvm
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_runtime
from tvm.contrib.debugger import debug_runtime

import os
import sys
sys.path.append('..')
import mixed_precision_models.quantized_resnet_v1 as quantized_resnet_v1
import mixed_precision_models.quantized_resnet_v1_5 as quantized_resnet_v1_5
from mixed_precision_models.layers import QConfig, QuantizeContext
import hawq_utils_resnet50

import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)

import argparse

parser = argparse.ArgumentParser(description='Mixed precision resnet example',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--tuning-enable', action='store_true', default=False,
                    help='Enable tuning')

parser.add_argument('--debug', action='store_true', default=False,
                    help='Debug')

parser.add_argument('--model-type', default='int4',
                    help='Mode datatype')

parser.add_argument('--batch-size', type=int, default=8,
                    help='Batch size')

parser.add_argument('--num-layers', type=int, default=50,
                    help='Number of layers in Resnet')

parser.add_argument('--version', default='1',
                    help='Resnet version')

parser.add_argument('--data-layout', default='HWNC',
                    help='Data layout (NHWC, NCWH)')

parser.add_argument('--kernel-layout', default='HWOI',
                    help='Kernel layout (OIHW, HWOI)')

parser.add_argument('--tuning-trials', type=int, default=50,
                    help='The length of tuining for each operation')

parser.add_argument('--with-bn', action='store_true', default=False,
                    help='Add batch normalization to the model')

parser.add_argument('--manual-code', action='store_true', default=False,
                    help='Use manual generated cuda code')

parser.add_argument('--with-softmax', action='store_true', default=False,
                    help='Use manual generated cuda code')

parser.add_argument('--debug-unit', default=None,
                    help='Debug specific unit input, compare the unit input to the pytorch result (stage1_unit1, stage1_unit2 ...)')

parser.add_argument('--bit-config', default=None,
                    help='Bit config to use')

args = parser.parse_args()

###############################################################################
# Set target device
# -----------------

TARGET_NAME = 'cuda'
CTX = tvm.context(TARGET_NAME, 0)

use_manual_code = args.manual_code
def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("debug_output"):
        os.mkdir("debug_output")
    write_code(code, "debug_output/resnet_generated.cu")
    if use_manual_code:
        code = open("debug_output/resnet_manual.cu").read()
    return code

###############################################################################
# Prepare Resnet model
# -----------------

batch_size = args.batch_size
image_shape = (3, 224, 224)
input_dtype = 'int8'
model_type = args.model_type
num_layers = args.num_layers
data_layout = args.data_layout
kernel_layout = args.kernel_layout

if num_layers == 18:
    stage = 4
    units = [2, 2, 2, 2]
    bottleneck = False
elif num_layers == 50:
    stage = 4
    units = [3, 4, 6, 3]
    bottleneck = True

if args.bit_config is not None:
    import bit_config
    hawq_utils_resnet50.load_qconfig_from_bit_config(stage, units, bit_config.bit_config_dict[args.bit_config], bottleneck)
else:
    if model_type == 'int4':
        int4_default_qconfig = QConfig(from_dtype='int32', from_scale=65.0, from_zero_point=0.0, input_dtype='uint4', input_scale=8.0, input_zero_point=0.0,
                                                kernel_dtype='int4', kernel_scale=8.0, kernel_zero_point=0.0, output_dtype='int32', output_scale=75.0, output_zero_point=0.0)

        QuantizeContext.set_default_qconfig(int4_default_qconfig)
        QuantizeContext.qconfig_dict = {
            "conv0_qconfig" : QConfig(from_dtype='int32', from_scale=65.0, from_zero_point=0.0, input_dtype='int8', input_scale=8.0, input_zero_point=0.0,
                                                kernel_dtype='int8', kernel_scale=8.0, kernel_zero_point=0.0, output_dtype='int32', output_scale=75.0, output_zero_point=0.0),
            #  "stage2_unit1_qconfig1" : QConfig(from_dtype='int32', from_scale=65.0, from_zero_point=0.0, input_dtype='int8', input_scale=8.0, input_zero_point=0.0,
            #                                     kernel_dtype='int8', kernel_scale=8.0, kernel_zero_point=0.0, output_dtype='int32', output_scale=65.0, output_zero_point=0.0),
            #  "stage3_unit1_qconfig1" : QConfig(from_dtype='int32', from_scale=65.0, from_zero_point=0.0, input_dtype='int8', input_scale=8.0, input_zero_point=0.0,
            #                                    kernel_dtype='int8', kernel_scale=8.0, kernel_zero_point=0.0, output_dtype='int32', output_scale=65.0, output_zero_point=0.0),
            # "stage4_unit1_qconfig1" : QConfig(from_dtype='int32', from_scale=64.0, from_zero_point=0.0, input_dtype='int8', input_scale=8.0, input_zero_point=0.0,
            #                                    kernel_dtype='int8', kernel_scale=8.0, kernel_zero_point=0.0, output_dtype='int32', output_scale=64.0, output_zero_point=0.0),

        }

        # units = [3, 4, 6, 3]
        # for stage in range(1):
        #     for unit in range(units[stage]-1):
        #         QuantizeContext.qconfig_dict["stage%d_unit%d_qconfig1" % (stage+1, unit+2)] = QConfig(from_dtype='int32', from_scale=64.0, from_zero_point=0.0, input_dtype='int8', input_scale=8.0, input_zero_point=0.0,
        #                                                                                             kernel_dtype='int8', kernel_scale=8.0, kernel_zero_point=0.0, output_dtype='int32', output_scale=64.0, output_zero_point=0.0)

def get_random_input():
    if data_layout == 'NCHW':
        data_shape = (batch_size, image_shape[0], image_shape[1], image_shape[2])
    elif data_layout == 'NHWC':
        data_shape = (batch_size, image_shape[1], image_shape[2], image_shape[0])
    elif data_layout == 'HWNC':
        # use NHWC input and transpose to HWNC in the network
        data_shape = (batch_size, image_shape[1], image_shape[2], image_shape[0])
    else:
        sys.exit(1)

    if input_dtype == 'float32':
        return np.random.uniform(size=data_shape).astype('float32')
    if input_dtype == 'int8':
        return np.random.randint(low=-8, high=8, size=data_shape).astype('int8')
    else:
        sys.exit(1)


input_data = get_random_input()
if args.version == '1.5':
    func, params = quantized_resnet_v1_5.get_workload(batch_size=batch_size,
                                                    image_shape=image_shape,
                                                    num_layers=num_layers,
                                                    dtype=input_dtype,
                                                    data_layout=data_layout,
                                                    kernel_layout=kernel_layout,
                                                    with_bn=args.with_bn,
                                                    rounding="UPWARD")
else:
    func, params = quantized_resnet_v1.get_workload(batch_size=batch_size,
                                                    image_shape=image_shape,
                                                    num_layers=num_layers,
                                                    dtype=input_dtype,
                                                    data_layout=data_layout,
                                                    kernel_layout=kernel_layout,
                                                    with_bn=args.with_bn,
                                                    rounding="UPWARD",
                                                    with_softmax=args.with_softmax,
                                                    debug_unit=args.debug_unit)

# print(func.astext())
###############################################################################
# Tuning
# -----------------
tuning_enable = args.tuning_enable
# log_filename = "./mixed_precision_models/tuning_logs/resnet%d_%s_%s_batch_%d.log" % (num_layers, data_layout, model_type, batch_size)
log_filename = "./mixed_precision_models/tuning_logs/resnet%d_%s_mixed_batch_%d.log" % (num_layers, data_layout, batch_size)
tmp_log_file = log_filename + '.temp'

if tuning_enable:
    print("Extracting tasks ...")

    with relay.build_config(opt_level=3):
        tasks = autotvm.task.extract_from_program(func, target=TARGET_NAME, params=params)

    print(tasks)

    measure_option = autotvm.measure_option(
        builder='local',
        runner=autotvm.LocalRunner(number=20, repeat=3, min_repeat_ms=150)
        # runner=autotvm.RPCRunner(
        #    'T4',  # change the device key to your key
        #    '0.0.0.0', 9190,
        #    number=20, repeat=3, min_repeat_ms=150),
        )

    for i, task in enumerate(reversed(tasks)):
        print(task)
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        tuner = autotvm.tuner.XGBTuner(task, feature_type='knob')
        num_trial = min(args.tuning_trials, len(task.config_space))
        tuner.tune(n_trial=num_trial,
                   measure_option=measure_option,
                   callbacks=[autotvm.callback.progress_bar(num_trial, prefix=prefix),
                              autotvm.callback.log_to_file(tmp_log_file)])

        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)

###############################################################################
# Build graph
# -----------------
DEBUG = args.debug

if not os.path.exists(log_filename):
    log_filename = None
else:
    print("Apply tuning log " + log_filename)

with autotvm.apply_history_best(log_filename):
    with relay.build_config(opt_level=3):

        graph, lib, params = relay.build(func, target=TARGET_NAME, params=params)

        if DEBUG:
            print("Debug mode")
            debug_dir = './debug_output'
            if not os.path.exists(debug_dir):
                os.mkdir(debug_dir)

            lib.export_library("./debug_output/net.tar")
            with open("./debug_output/resnet.cu", "w") as source_file:
                source_file.write(lib.imported_modules[0].get_source())

            m = tvm.contrib.debugger.debug_runtime.create(graph, lib, CTX, dump_root=debug_dir)

            # Set the network parameters and inputs
            m.set_input(**params)
            m.set_input('data', input_data)

            m.run()
        else:
            module = tvm.contrib.graph_runtime.create(graph, lib, ctx=CTX)

            module.set_input("data", input_data)
            module.set_input(**params)

            # warm up
            for i in range(0, 100):
                module.run()
                module.get_output(0).asnumpy()

            num = 50  # number of times we run module for a single measurement
            rep = 30  # number of measurements (we derive std dev from this)
            timer = module.module.time_evaluator("run", CTX, number=num, repeat=rep)

            tcost = timer()
            std = np.std(tcost.results) * 1000
            mean = tcost.mean * 1000

            print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, batch_size))
            print("Average per sample inference time: %.2fms" % (mean/batch_size))

    # for var, qconfig in QuantizeContext.qconfig_print.items():
    #     print("\'" + var + "\'" + " : " + str(qconfig) + ',')
