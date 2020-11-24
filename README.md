# HAWQ: Hessian AWare Quantization

HAWQ is an advanced quantization library written for PyTorch. HAWQ enables low-precision and mixed-precision uniform quantization, with direct hardware implementation through TVM.

## Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install HAWQ** and develop locally:
```bash
git clone https://github.com/zhendong/HAWQ
```

## Quick Start
### Quantization-Aware Training
An example to run uniform 8-bit quantization for resnet50 on ImageNet. 
```
export CUDA_VISIBLE_DEVICES=0; 
python quant_train.py -a resnet50 --epochs 90 --lr 0.0001 --batch-size 128 --data /path/to/imagenet/ --pretrained --save-path /path/to/checkpoints/ --act-range-momentum=0.99 --wd 1e-4 --data-percentage 0.0001 --fold-BN 1 --fix-BN 1 --checkpoint-iter -1 --quant-scheme uniform8
```


## Related Works
  - [HAWQ-V3: Dyadic Neural Network Quantization](https://arxiv.org/abs/2011.10680)
  - [HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks (NeurIPS 2020)](https://arxiv.org/abs/1911.03852)
  - [HAWQ: Hessian AWare Quantization of Neural Networks With Mixed-Precision (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html)


## License

HAWQ is released under the [Apache 2.0 license](LICENSE).
