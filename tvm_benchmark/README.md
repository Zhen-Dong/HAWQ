# Instructions To Run The iPython Notebook

## Google Cloud Environment Set Up

* In the Google Cloud console, click on **Go to Compute Engine**.
* Click **Create Instance**.
* Select a region with GPU resources available, e.g. us-west1 (Oregon).
* For Machine configuration, select **General-purpose** for Machine family, **N1** for Series, and **n1-standard-4 (4 vCPU, 15 GB memory)** for Machine type.
* Click on **CPU platform and GPU** for more options.
* For CPU platform, select **Intel Skylake or later**.
* Click on **Add GPU** and select **NVIDIA Tesla T4**
* For the Boot disk, under Public images, select **Deep Learning on Linux** for Operating system, Deep Learning Image: **PyTorch 1.4.0 and fastai m49** for Version, **Standard persistent disk** for Boot disk type, and a reasonable disk size (e.g. 128GB).
* Under Identity and API access, under Access scopes, select **Allow full access to all Cloud APIs**.
* Under Firewall, Check both the **Allow HTTP traffic** and the **Allow HTTPS traffic** boxes.
* When you first SSH into the VM, if asked to install Nvidia drivers, say yes.

## Installing zachzzc's TVM

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


## Instructions to run TVM inference and debug it
1. Create a new directory named "data" under tvm_test.

2. Download the reference pytorch data from https://drive.google.com/drive/folders/1gkhqaeklP0n8QS_72q0YOrbbEw2OF3Sz?usp=sharing into the newly created folder. The files can each be downloaded directly from the Google Drive into the Google Cloud VM using `wget` by following the steps [here](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99):
* Right click on the file to download and then choose **Share**.
* Make sure the share options are set to **Anyone on the internet with this link can view**. Press **Copy link** and paste it somewhere to view.
* Run the following command, replacing FILEID with the id in the shared link and FILENAME with the name of the file.
~~~~
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt
~~~~
For example if the sharing link is https://drive.google.com/file/d/1h0kC7NfTsi0XaBmupMQqZVpDWEh-euE0/view?usp=sharing, then 1h0kC7NfTsi0XaBmupMQqZVpDWEh-euE0 is the FILEID, featuremaps.pth.tar is the FILENAME, and the command to download the file would be
~~~~
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1h0kC7NfTsi0XaBmupMQqZVpDWEh-euE0' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1h0kC7NfTsi0XaBmupMQqZVpDWEh-euE0" -O featuremaps.pth.tar && rm -rf /tmp/cookies.txt
~~~~

3. Convert the Pytorch parameters into TVM format by running
~~~~
    python hawq_utils.py --model-dir ./data
~~~~
* It should give a key error KeyError: 'module.features.stage3.unit1.block_input_featuremap'. It is fine because for now we only save the checkpoints up to stage2.

4. There are two modes to run the TVM inference
* Normal mode: Run the inference and make predictions of the given image
~~~~
    python test_resnet_inference.py --model-dir ./data
~~~~

* Debug mode: Run the inference until the specific unit. The result will be compared with the Pytorch checkpoint and save under "./data/tvm_result". The debug unit can be stage1_unit1_input, stage1_unit1_output ... fc_input, fc_output
~~~~
    python test_resnet_inference.py --model-dir ./data --debug-unit "stage2_unit2_input"
~~~~

5. Run Imagenet accuracy test
~~~~
    python test_resnet_accuracy_imagenet.py --model-dir ./data --val-dir PATH_TO_IMAGENET_VALIDATION_DATASET
~~~~

6. Measure inference time (with uniform int4/int8 or custom bit config in bit_config.py)
- With uniform int4
~~~~
    python test_resnet_inference_time.py --num-layers 50 --batch-size 8 --data-layout "HWNC" --model-type "int4"
~~~~
- Custom bit config in bit_config.py
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
- Driver script to automatically optimize int4 casting for tvm auto-generated CUDA codes
~~~~
    ./run_restnet_inference_time.sh
~~~~


