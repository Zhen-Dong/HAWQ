<p align="center">
  <img src="imgs/resnet18_TC.png" width="150">
  <br />
  <br />
  </p>

# HAWQ: Hessian AWare Quantization

HAWQ is an advanced quantization library written for PyTorch. HAWQ enables low-precision and mixed-precision uniform quantization, with direct hardware implementation through TVM.

For more details please see:

- [HAWQ-V3 lightning talk in TVM Conference](https://www.youtube.com/watch?v=VRiujqKU254)
- [HAWQ-V2 presentation in NeurIPS'20](https://neurips.cc/virtual/2020/public/poster_d77c703536718b95308130ff2e5cf9ee.html)

## Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* **To install HAWQ** and develop locally:
```bash
git clone https://github.com/zhendong/HAWQ.git
cd HAWQ
pip install -r requirements.txt
```

## Quick Start
### Quantization-Aware Training
An example to run uniform 8-bit quantization for resnet50 on ImageNet. 
```
export CUDA_VISIBLE_DEVICES=0
python quant_train.py -a resnet50 --epochs 90 --lr 0.0001 --batch-size 128 --data /path/to/imagenet/ --pretrained --save-path /path/to/checkpoints/ --act-range-momentum=0.99 --wd 1e-4 --data-percentage 0.0001 --fold-BN 1 --fix-BN 1 --checkpoint-iter -1 --quant-scheme uniform8
```

### Inference Acceleration
* [Instructions on Hardware Implementation through TVM](tvm_benchmark/README.md)

## Related Works
  - [HAWQ-V3: Dyadic Neural Network Quantization](https://arxiv.org/abs/2011.10680)
  - [HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks (NeurIPS 2020)](https://arxiv.org/abs/1911.03852)
  - [HAWQ: Hessian AWare Quantization of Neural Networks With Mixed-Precision (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Dong_HAWQ_Hessian_AWare_Quantization_of_Neural_Networks_With_Mixed-Precision_ICCV_2019_paper.html)


## License

HAWQ is released under the [MIT license](LICENSE).
