# Model Zoo

## ResNet18 on ImageNet 
Model | Quantization | Model Size(MB) | BOPS(G) | Accuracy(%) | Download
---|---|---|---|---|---
`ResNet18` | Floating Points | 44.6 | 1858 | 71.47 | [resnet18_baseline](https://drive.google.com/drive/folders/1C2EpDnRkCFwrH5drwLQRDmlsiQNKN9Nc?usp=sharing)
`ResNet18` | W8A8            | 11.1 | 116  | 71.56 | [resnet18_uniform8](https://drive.google.com/drive/folders/1BbJfgkzGVyPNlXrAO9TbZrVQMqFt99JM?usp=sharing)
`ResNet18` | Mixed Precision | 6.7  | 72   | 70.22 | [resnet18_mp]()
`ResNet18` | W4A4            | 5.8  | 34   | 68.45 | [resnet18_uniform4]()

## ResNet50 on ImageNet
Model | Quantization | Model Size(MB) | BOPS(G) | Accuracy(%) | Download
---|---|---|---|---|---
`ResNet50` | Floating Points | 97.8 | 3951 | 77.72 | [resnet50_baseline](https://drive.google.com/drive/folders/1C-R8gM2HF8sKi6MPJibopeb8JGklcVUW?usp=sharing)
`ResNet50` | W8A8            | 24.5 | 247  | 77.58 | [resnet50_uniform8](https://drive.google.com/drive/folders/19xmwcVJzJANGagCESZAcwAJbIhRBJy4J?usp=sharing)
`ResNet50` | Mixed Precision | 18.7 | 154  | 75.39 | [resnet50_mp]()
`ResNet50` | W4A4            | 13.1 | 67   | 74.24 | [resnet50_uniform4]()

Baseline models are from [PyTorchCV](https://pypi.org/project/pytorchcv/).

## Download Quantized Models
The files can be downloaded directly from the Google Drive. \
Optionally you can use `wget` by following the steps [here](https://medium.com/@acpanjan/download-google-drive-files-using-wget-3c2c025a8b99):
* Press **Copy link** and paste it somewhere to view.
* Run the following command, replacing FILEID with the id in the shared link and FILENAME with the name of the file.
~~~~
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt
~~~~
For example if the sharing link is https://drive.google.com/drive/folders/1BbJfgkzGVyPNlXrAO9TbZrVQMqFt99JM, then 1BbJfgkzGVyPNlXrAO9TbZrVQMqFt99JM is the FILEID, resnet18_uniform8 is the FILENAME, and the command to download the file would be:
~~~~
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BbJfgkzGVyPNlXrAO9TbZrVQMqFt99JM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BbJfgkzGVyPNlXrAO9TbZrVQMqFt99JM" -O resnet18_uniform8 && rm -rf /tmp/cookies.txt
~~~~
