# Model Zoo

## ResNet18 on ImageNet 
Model | Quantization | Model Size(MB) | BOPS(G) | Accuracy(%) | Download
---|---|---|---|---|---
`ResNet18` | Floating Points | 44.6 | 1858 | 71.47 | [resnet18_baseline](https://drive.google.com/file/d/1C7is-QOiSlLXKoPuKzKNxb0w-ixqoOQE/view?usp=sharing)
`ResNet18` | W8A8            | 11.1 | 116  | 71.56 | [resnet18_uniform8](https://drive.google.com/file/d/1CLAd3LhiRVYwiBZRuUJgrzrrPFfLvfWG/view?usp=sharing)
`ResNet18` | Mixed Precision | 6.7  | 72   | 70.22 | [resnet18_mp]()
`ResNet18` | W4A4            | 5.8  | 34   | 68.45 | [resnet18_uniform4]()

## ResNet50 on ImageNet
Model | Quantization | Model Size(MB) | BOPS(G) | Accuracy(%) | Download
---|---|---|---|---|---
`ResNet50` | Floating Points | 97.8 | 3951 | 77.72 | [resnet50_baseline](https://drive.google.com/file/d/1CE4b05gwMzDqcdpwHLFC2BM0841qKJp8/view?usp=sharing)
`ResNet50` | W8A8            | 24.5 | 247  | 77.58 | [resnet50_uniform8](https://drive.google.com/file/d/1CID7aId-SL8edGx8j5-Lsup_GqW3OX7-/view?usp=sharing)
`ResNet50` | Mixed Precision | 18.7 | 154  | 75.39 | [resnet50_mp]()
`ResNet50` | W4A4            | 13.1 | 67   | 74.24 | [resnet50_uniform4]()

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
