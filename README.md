# Rectal Cancer LNM Classification
## Introduction
We design an efficient classification strategy for rectal cancer LNM classification task. Specifically, we design a sub-patch strategy for data augmentation and use Med3D pre-trained ResNet as the model backbone.

## Usage
### Installation
1. Requirements

- Linux
- Python 3.7+
- PyTorch 1.10.0 or higher
- MONAI 0.8.1
- CUDA 10.0 or higher

2. Install dependencies.
```shell
pip install -r requirements.txt
```
### Dataset
After preprocessing, the tumor is cropped and saved slice by slice as .png images.
We provide several preprocessed sample data. See [here](./sample_data).
The preprocessing code that transfers Dicom files to PNG images will be uploaded later.

### Training and Evaluation

```
python Med3D_train.py
```
- If `-pretrained` is set to True, you need to also set `-pretrained_path` of [Med3D](https://github.com/Tencent/MedicalNet).

