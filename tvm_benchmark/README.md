# Instructions on Hardware Deployment
In this README, we show the steps to run quantized models on NVIDIA T4 GPU for inference speed-up.
## Google Cloud Environment Set Up

* In the Google Cloud console, click on **Go to Compute Engine**.
* Click **Create Instance**.
* Select a region with GPU resources available, e.g. us-west1 (Oregon).
* For Machine configuration, select **General-purpose** for Machine family, **N1** for Series, and **n1-standard-4 (4 vCPU, 15 GB memory)** for Machine type.
* Click on **CPU platform and GPU** for more options.
* For CPU platform, select **Intel Skylake or later**.
* Click on **Add GPU** and select **NVIDIA Tesla T4**
* For the Boot disk, under Public images, select **Deep Learning on Linux** for Operating system, Deep Learning Image: **PyTorch 1.4.0 and fastai m49** for Version, **Standard persistent disk** for Boot disk type, and a reasonable disk size (e.g. 256GB).
* Under Identity and API access, under Access scopes, select **Allow full access to all Cloud APIs**.
* Under Firewall, Check both the **Allow HTTP traffic** and the **Allow HTTPS traffic** boxes.
* When you first SSH into the Virtual Machine (VM), if asked to install Nvidia drivers, type yes.

## Install zachzzc's TVM

A lot of these steps are taken from the TVM [Install from Source](https://tvm.apache.org/docs/install/from_source.html) page.

Clone and checkout the specific branch in the following repo.

    git clone --recursive --branch int4_direct_HWNC http://github.com/zachzzc/incubator-tvm.git ~/tvm

Install LLVM.

    sudo bash -c "$(wget -O - http://apt.llvm.org/llvm.sh)"

Install build dependencies.

    sudo apt-get update
    sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev

Install CUDA 10.2 (needed for uint4 support). Choose yes for all the prompts.

    wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
    sudo sh cuda_10.2.89_440.33.01_linux.run

Change directory to the TVM directory, create a build directory and copy `cmake/config.cmake` to the directory.

    cd ~/tvm
    mkdir build
    cp cmake/config.cmake build

Edit `build/config.cmake` to customize compilation options:

* Change `set(USE_CUDA OFF)` to `set(USE_CUDA ON)` to enable CUDA.
* Change `set(USE_LLVM OFF)` to `set(USE_LLVM /path/to/llvm-config)` to build with LLVM. The `/path/to/llvm-config` should be something like `/usr/lib/llvm-10/bin/llvm-config`.
* Optional: for additional debugging options, change `set(USE_GRAPH_RUNTIME_DEBUG OFF)` to `set(USE_GRAPH_RUNTIME_DEBUG ON)` and `set(USE_RELAY_DEBUG OFF)` to `set(USE_RELAY_DEBUG ON)`.

Build TVM and related libraries.

    cd build
    cmake ..
    make -j4

Install the TVM Python package by appending the following lines to *~/.bashrc*:

    export TVM_HOME=/path/to/tvm
    export PYTHONPATH=$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}

and then `source ~/.bashrc` to apply the changes.

Install Python dependencies.
* Necessary dependencies: `pip3 install --user numpy decorator attrs`
* For RPC Tracker: `pip3 install --user tornado`
* For the auto-tuning module: `pip3 install --user tornado psutil xgboost`


## Run TVM inference
1. Create a new directory named "data" under tvm_test.

2. Conduct quantization-aware training (QAT) to get quantized model and put it in the "data" directory. \
Or download the quantized PyTorch model from [model zoo](../model_zoo.md) into the newly created folder. 
Optionally you can use `wget` by following the steps [here](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99):
* Press **Copy link** and paste it somewhere to view.
* Run the following command, replacing FILEID with the id in the shared link and FILENAME with the name of the file.
~~~~
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt
~~~~
For example if the sharing link is https://drive.google.com/file/d/1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE/view?usp=sharing, then 1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE is the FILEID, resnet18_uniform8.tar.gz is the FILENAME, and the command to download the file would be:
~~~~
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE" -O resnet18_uniform8.tar.gz && rm -rf /tmp/cookies.txt
~~~~

3. Convert the PyTorch parameters into TVM format by running the following command. Model directory contains the checkpoint.pth.tar file.
* Normal mode: (no comparison with featuremaps, only convert weights and biases)
~~~~
    python hawq_utils_resnet50.py --model-dir ./data
~~~~
* Debug mode: (including comparison with PyTorch featuremaps)
~~~~
    python hawq_utils_resnet50.py --model-dir ./data --with-featuremap
~~~~

4. Run TVM inference on a single image. To run debug mode, we need to store the image and featuremaps.
* Normal mode: (Run the inference and make predictions of the given image)
~~~~
    python test_resnet_inference.py --model-dir ./data
~~~~
* Debug mode: (Run inference until the specific unit. The results will be compared with PyTorch featuremaps and saved under "./data/tvm_result". The debug unit can be set to stage1_unit1_input, stage1_unit1_output ... fc_input, fc_output)
~~~~
    python test_resnet_inference.py --model-dir ./data --debug-unit "stage2_unit2_input"
~~~~

5. Run accuracy test on ImageNet
~~~~
    python test_resnet_accuracy_imagenet.py --model-dir ./data --val-dir PATH_TO_IMAGENET_VALIDATION_DATASET
~~~~

6. Measure inference time (with uniform int4/int8 or custom mixed-precision bit configs in bit_config.py).
- With uniform int4 quantized model
~~~~
    python test_resnet_inference_time.py --num-layers 50 --batch-size 8 --data-layout "HWNC" --model-type "int4"
~~~~
- With a custom bit config in bit_config.py
~~~~
    python test_resnet_inference_time.py --num-layers 50 --batch-size 8 --data-layout "HWNC" --bit-config BIT_CONFIG_NAME
~~~~
- With manual optimized CUDA code
~~~~
    python test_resnet_inference_time.py --num-layers 50 --batch-size 8 --data-layout "HWNC" --bit-config BIT_CONFIG_NAME --manual-code
~~~~
- Generate layerwise breakdown
~~~~
    python test_resnet_inference_time.py --num-layers 50 --batch-size 8 --data-layout "HWNC" --model-type "int4" --debug
~~~~
- Run driver script to automatically optimize int4 casting for tvm-generated CUDA codes
~~~~
    ./run_restnet_inference_time.sh
~~~~


