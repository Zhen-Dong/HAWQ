# Model Zoo

## ResNet18 on ImageNet 
Model | Quantization Scheme | Model Size(MB) | BOPS(G) | Speed-Up | Accuracy(%) | Download
---|---|---|---|---|---|---
`ResNet18` | Floating Points                 | 44.6 | 1858 | NA    | 71.47 | [resnet18_baseline](https://drive.google.com/file/d/1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE/view?usp=sharing)
`ResNet18` | W8A8                            | 11.1 | 116  | 1.00x    | 71.56 | [resnet18_uniform8](https://drive.google.com/file/d/1CLAd3LhiRVYwiBZRuUJgrzrrPFfLvfWG/view?usp=sharing)
`ResNet18` | Mixed Precision High Size       | 9.9  | 103  | 1.03x | 71.20 | [resnet18_size0.75]()
`ResNet18` | Mixed Precision Medium Size     | 7.9  | 98   | 1.06x | 70.50 | [resnet18_size0.5]()
`ResNet18` | Mixed Precision Low Size        | 7.3  | 95   | 1.08x | 70.01 | [resnet18_size0.25]()
`ResNet18` | Mixed Precision High BOPS       | 8.7  | 92   | 1.12x | 70.40 | [resnet18_bops0.75]()
`ResNet18` | Mixed Precision Medium BOPS     | 6.7  | 72   | 1.21x | 70.22 | [resnet18_bops0.5](https://drive.google.com/file/d/1DbDXYdulvvb9YOG1fRSrCVPvry_Reu8z/view?usp=sharing)
`ResNet18` | Mixed Precision Low BOPS        | 6.1  | 54   | 1.35x | 68.72 | [resnet18_bops0.25]()
`ResNet18` | Mixed Precision High Latency    | 8.7  | 92   | 1.12x | 70.40 | [resnet18_latency0.75]()
`ResNet18` | Mixed Precision Medium Latency  | 7.2  | 76   | 1.19x | 70.34 | [resnet18_latency0.5]()
`ResNet18` | Mixed Precision Low Latency     | 6.1  | 54   | 1.35x | 68.56 | [resnet18_latency0.25]()
`ResNet18` | W4A4                            | 5.8  | 34   | 1.48x | 68.45 | [resnet18_uniform4](https://drive.google.com/file/d/1D4DPcW2s9QmSnKzUgcjH-2eYO8zpDRIL/view?usp=sharing)

## ResNet50 on ImageNet
Model | Quantization Scheme | Model Size(MB) | BOPS(G) | Speed-Up | Accuracy(%) | Download
---|---|---|---|---|---|---
`ResNet50` | Floating Points                 | 97.8 | 3951 | NA    | 77.72 | [resnet50_baseline](https://drive.google.com/file/d/1CE4b05gwMzDqcdpwHLFC2BM0841qKJp8/view?usp=sharing)
`ResNet50` | W8A8                            | 24.5 | 247  | 1.00x    | 77.58 | [resnet50_uniform8](https://drive.google.com/file/d/1CID7aId-SL8edGx8j5-Lsup_GqW3OX7-/view?usp=sharing)
`ResNet50` | Mixed Precision High Size       | 21.3 | 226  | 1.09x | 77.38 | [resnet50_size0.75]()
`ResNet50` | Mixed Precision Medium Size     | 19.0 | 197  | 1.13x | 75.95 | [resnet50_size0.5]()
`ResNet50` | Mixed Precision Low Size        | 16.0 | 168  | 1.18x | 74.89 | [resnet50_size0.25]()
`ResNet50` | Mixed Precision High BOPS       | 22.0 | 197  | 1.16x | 76.10 | [resnet50_bops0.75]()
`ResNet50` | Mixed Precision Medium BOPS     | 18.7 | 154  | 1.23x | 75.39 | [resnet50_bops0.5](https://drive.google.com/file/d/1DNUkyavD10saZw9_7TzJhEy0NFPhSVZr/view?usp=sharing)
`ResNet50` | Mixed Precision Low BOPS        | 16.7 | 110  | 1.30x | 74.45 | [resnet50_bops0.25]()
`ResNet50` | Mixed Precision High Latency    | 22.3 | 199  | 1.13x | 76.63 | [resnet50_latency0.75]()
`ResNet50` | Mixed Precision Medium Latency  | 18.5 | 155  | 1.21x | 74.95 | [resnet50_latency0.5]()
`ResNet50` | Mixed Precision Low Latency     | 16.5 | 114  | 1.28x | 74.26 | [resnet50_latency0.25]()
`ResNet50` | W4A4                            | 13.1 | 67   | 1.45x | 74.24 | [resnet50_uniform4](https://drive.google.com/file/d/1DDis-8C-EupCRj-ExH58ldSv-tG2RXyf/view?usp=sharing)

## ResNet101 on ImageNet
Model | Quantization | Model Size(MB) | BOPS(G) | Accuracy(%) | Download
---|---|---|---|---|---
`ResNet101` | Floating Points | 170.0 | 7780 | 78.10 | [resnet101_baseline]()
`ResNet101b` | Floating Points | 170.0 | 8018 | 79.41 | [resnet101b_baseline]()

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
