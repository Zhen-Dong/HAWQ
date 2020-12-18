# Model Zoo

## ResNet18 on ImageNet 
Model | Quantization Scheme | Model Size(MB) | BOPS(G) | Speed-Up | Accuracy(%) | Download
---|---|---|---|---|---|---
`ResNet18` | Floating Points                 | 44.6    | 1858   | NA        | 71.47 | [resnet18_baseline](https://drive.google.com/file/d/1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE/view?usp=sharing)
`ResNet18` | W8A8                            | 11.1    | 116    | 1.00x     | 71.56 | [resnet18_uniform8](https://drive.google.com/file/d/1CLAd3LhiRVYwiBZRuUJgrzrrPFfLvfWG/view?usp=sharing)
`ResNet18` | Mixed Precision High Size       | **9.9** | 103    | 1.03x     | 71.20 | [resnet18_size0.75](https://drive.google.com/file/d/1Fjm1Wruo773e3-jTIGahUWWQbmqGyMLO/view?usp=sharing)
`ResNet18` | Mixed Precision Medium Size     | **7.9** | 98     | 1.06x     | 70.50 | [resnet18_size0.5](https://drive.google.com/file/d/1EGH76MRLckRtRXqWZHJ_I5DW5UQ-C8iA/view?usp=sharing)
`ResNet18` | Mixed Precision Low Size        | **7.3** | 95     | 1.08x     | 70.01 | [resnet18_size0.25](https://drive.google.com/file/d/1Eq9tmF8XlxOQGNMOuvc5rTPV0N4Ov-4C/view?usp=sharing)
`ResNet18` | Mixed Precision High BOPS       | 8.7     | **92** | 1.12x     | 70.40 | [resnet18_bops0.75](https://drive.google.com/file/d/1F-pcK-AMCNcPAOydmhJN5aiDGGaEk-q7/view?usp=sharing)
`ResNet18` | Mixed Precision Medium BOPS     | 6.7     | **72** | 1.21x     | 70.22 | [resnet18_bops0.5](https://drive.google.com/file/d/1DbDXYdulvvb9YOG1fRSrCVPvry_Reu8z/view?usp=sharing)
`ResNet18` | Mixed Precision Low BOPS        | 6.1     | **54** | 1.35x     | 68.72 | [resnet18_bops0.25](https://drive.google.com/file/d/1G9UgvLB3KuDyqNj4xV7DFiHjfXtULPJI/view?usp=sharing)
`ResNet18` | Mixed Precision High Latency    | 8.7     | 92     | **1.12x** | 70.40 | [resnet18_latency0.75](https://drive.google.com/file/d/1FcDVQT-p314lDq-URbHbLCSkGnWrd_vT/view?usp=sharing)
`ResNet18` | Mixed Precision Medium Latency  | 7.2     | 76     | **1.19x** | 70.34 | [resnet18_latency0.5](https://drive.google.com/file/d/1EfpPjgx-q5IS9rDP1irrdQtMvBodkDei/view?usp=sharing)
`ResNet18` | Mixed Precision Low Latency     | 6.1     | 54     | **1.35x** | 68.56 | [resnet18_latency0.25](https://drive.google.com/file/d/1FwC7Sjp9lFW6dLdnyb9O4Re7OLkUpkPy/view?usp=sharing)
`ResNet18` | W4A4                            | 5.8     | 34     | 1.48x     | 68.45 | [resnet18_uniform4](https://drive.google.com/file/d/1D4DPcW2s9QmSnKzUgcjH-2eYO8zpDRIL/view?usp=sharing)

## ResNet50 on ImageNet
Model | Quantization Scheme | Model Size(MB) | BOPS(G) | Speed-Up | Accuracy(%) | Download
---|---|---|---|---|---|---
`ResNet50` | Floating Points                 | 97.8     | 3951     | NA        | 77.72 | [resnet50_baseline](https://drive.google.com/file/d/1CE4b05gwMzDqcdpwHLFC2BM0841qKJp8/view?usp=sharing)
`ResNet50` | W8A8                            | 24.5     | 247      | 1.00x     | 77.58 | [resnet50_uniform8](https://drive.google.com/file/d/1CID7aId-SL8edGx8j5-Lsup_GqW3OX7-/view?usp=sharing)
`ResNet50` | Mixed Precision High Size       | **21.3** | 226      | 1.09x     | 77.38 | [resnet50_size0.75](https://drive.google.com/file/d/1GtYgWFQrWfmn-23pFrZlxmBtuDCRG5Zs/view?usp=sharing)
`ResNet50` | Mixed Precision Medium Size     | **19.0** | 197      | 1.13x     | 75.95 | [resnet50_size0.5](https://drive.google.com/file/d/1DnnRL9Q9SJ6BA5M98zGxcKrrAKClDdfJ/view?usp=sharing)
`ResNet50` | Mixed Precision Low Size        | **16.0** | 168      | 1.18x     | 74.89 | [resnet50_size0.25](https://drive.google.com/file/d/1H_rLcaOobHCASSxLD5F6ho5rKNvBqAOo/view?usp=sharing)
`ResNet50` | Mixed Precision High BOPS       | 22.0     | **197**  | 1.16x     | 76.10 | [resnet50_bops0.75](https://drive.google.com/file/d/1H5947bedQ1rCGzdKpSCJjIxysJUBznOE/view?usp=sharing)
`ResNet50` | Mixed Precision Medium BOPS     | 18.7     | **154**  | 1.23x     | 75.39 | [resnet50_bops0.5](https://drive.google.com/file/d/1DNUkyavD10saZw9_7TzJhEy0NFPhSVZr/view?usp=sharing)
`ResNet50` | Mixed Precision Low BOPS        | 16.7     | **110**  | 1.30x     | 74.45 | [resnet50_bops0.25](https://drive.google.com/file/d/1G_JQJgGTDYQN5atmcyjDsJZV5zkH8GWw/view?usp=sharing)
`ResNet50` | Mixed Precision High Latency    | 22.3     | 199      | **1.13x** | 76.63 | [resnet50_latency0.75](https://drive.google.com/file/d/1HBQhrTplhOHft43WEifaq35dfUftP5tJ/view?usp=sharing)
`ResNet50` | Mixed Precision Medium Latency  | 18.5     | 155      | **1.21x** | 74.95 | [resnet50_latency0.5](https://drive.google.com/file/d/1GbviN74Z806jyDusohusEjgKuqIyAc5s/view?usp=sharing)
`ResNet50` | Mixed Precision Low Latency     | 16.5     | 114      | **1.28x** | 74.26 | [resnet50_latency0.25](https://drive.google.com/file/d/1HuMaFhL1GV3XiYt9fLncZf6QruL7eGif/view?usp=sharing)
`ResNet50` | W4A4                            | 13.1     | 67       | 1.45x     | 74.24 | [resnet50_uniform4](https://drive.google.com/file/d/1DDis-8C-EupCRj-ExH58ldSv-tG2RXyf/view?usp=sharing)

## ResNet101 on ImageNet
Model | Quantization | Model Size(MB) | BOPS(G) | Accuracy(%) | Download
---|---|---|---|---|---
`ResNet101` | Floating Points | 170.0 | 7780 | 78.10 | [resnet101_baseline](https://drive.google.com/file/d/1GDliS9_HdQ75eH2G1bCivfQL4FTI4euF/view?usp=sharing)
`ResNet101b` | Floating Points | 170.0 | 8018 | 79.41 | [resnet101b_baseline](https://drive.google.com/file/d/1GPBBpi3wAko4wHWt8ZBOJj7RALW1_K1p/view?usp=sharing)

Baseline models are from [PyTorchCV](https://pypi.org/project/pytorchcv/).

## Download Quantized Models
The files can be downloaded directly from the Google Drive. \
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

## Commands and Notes
To conduct quantization-aware training, run the following command with correct attributes. \
Specifically, network architecture should be in the PyTorchCV form (resnet18, resnet50, etc), quantization scheme should correspond to names in the bit_config.py file (uniform8, size0.5, etc).
```
export CUDA_VISIBLE_DEVICES=0
python quant_train.py -a resnet50 --epochs 1 --lr 0.0001 --batch-size 128 --data /path/to/imagenet/ --pretrained --save-path /path/to/checkpoints/ --act-range-momentum=0.99 --wd 1e-4 --data-percentage 0.0001 --fix-BN --checkpoint-iter -1 --quant-scheme uniform8
```
Notes:
* 8-bit quantization-aware training typically converges fast, so the data-percentage attribute and the epoch attribute can be set to small values to only use a subset of training data with small number of epochs. For other cases with more aggressive quantization, the data-percentage should be set to 0.1 or 1, and the number of epochs should be adjusted (typical value is 90). Some examples for (quantization scheme : data-percentage): (size0.75 : 0.01); (bops0.75/latency0.75 : 0.1); (size0.5/bops0.5/latency0.5 : 1).
* The activation percentile function can be helpful for some scenarios, but it is time-consuming since the PyTorch torch.topk function is relatively slow.
* The fix-BN attribute is for training, BN will always be folded and fixed during validation.
* This codebase is specialized for easy deployment on hardware, meaning sometimes it sacrifices accuracy for simpler operations. It uses standard symmetric channel-wise quantization for weights, asymmetric layer-wise quantization for activations (except for 8-bit, the hardware support only allow symmetric quantization for 8-bit).
* It is difficult for current hardware to efficiently support operations with 2bit, 3bit, 5bit, 6bit or 7bit, so this codebase uses 4-bit and 8-bit for mixed-precision quantization (as in HAWQV3). The mixed-precision with 2 ~ 8bit in HAWQV2 is asymmetric (fixed-point based) quantization, which uses layerwise quantization-aware training. The easiest way for now to reproduce HAWQV2 is [Distiller](https://github.com/IntelLabs/distiller), but this will not lead to accelerated inference.
* The provided quantized models typically have a small variation on accuracy (mostly higher) compared with those in the result table. These models are trained with a standard setting, and further accuracy improvement can be obtained by finding better schemes of quantization-aware training.

To evaluate the quantized model, use the following command and adjust the attributes accordingly:
```
# Directly running these commands will get 75.58% Top-1 Accuracy on ImageNet.
export CUDA_VISIBLE_DEVICES=0
python quant_train.py -a resnet50 --epochs 90 --lr 0.0001 --batch-size 128 --data /path/to/imagenet/ --save-path /path/to/checkpoints/ --act-range-momentum=0.99 --wd 1e-4 --data-percentage 1 --checkpoint-iter -1 --quant-scheme bops_0.5 --resume /path/to/resnet50_bops0.5/checkpoint.pth.tar --resume-quantize -e
```
To resume quantization-aware training from the quantized checkpoint, remove the -e attribute.
