# Extending Swin-UMamba for 3D Aortic Segmentation in CT Images with 3D-Selective-Scan Mechanism

This repository contains the implementation of **3D-SwinUMamba**, an advanced architecture for 3D aortic segmentation in CT images. The model extends the original [UMamba](https://github.com/bowang-lab/U-Mamba) and [Swin-UMamba](https://github.com/path-to-swin-umamba-repo) by incorporating a **3D-Selective-Scan (SS3D)** mechanism, enabling efficient processing of volumetric data through six distinct scanning paths. This repository is specifically for the [**AortaSeg2024 Challenge**](https://aortaseg24.grand-challenge.org/dataset-access-information/).

## Key Features:
- **3D Extension of Swin-UMamba**: Incorporates depth-wise scanning for improved volumetric segmentation in medical imaging.
- **3D-Selective-Scan (SS3D)**: Enhances long-range dependency modeling by processing patches along depth, height, and width axes in forward and reverse directions.
- **Region-Based Loss Function**: Designed to enhance segmentation performance by aggregating the 23 aortic labels into four distinct anatomical regions.

## Region-Based Loss Function:
This model employs a **region-based loss function** designed to enhance segmentation performance by aggregating the 23 aortic labels into four distinct anatomical regions:

- **Ascending and Arch Aorta** (labels 1–6)
- **Descending Aorta** (labels 7–9)
- **Abdominal Aorta** (labels 10–17)
- **Iliac Arteries and Peripheral Branches** (labels 18–23)

The region loss is combined with the traditional cross-entropy and dice loss applied to individual targets. This aggregation reduces label imbalance and enhances the stability of the training process.


## Installation 
The code implementation is based on U-Mamba [U-Mamba with nnUNet version 2.1.1](https://wanglab.ai/u-mamba.html)
Requirements: `Ubuntu 20.04/22.04`, `CUDA 11.8`
1. Create a virtual environment: `conda create -n autopet python=3.10 -y` and `conda activate autopet `
2. Install [Pytorch](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4) 2.4.0+cu118: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
3. You may need to install cuda 11.8 at the OS level. (Optional)
```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
$ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
$ sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda
```
4. Install [Mamba](https://github.com/state-spaces/mamba): `pip install causal-conv1d>=1.2.0` and `pip install mamba-ssm --no-cache-dir`
5. Download code: `git clone https://github.com/BBQtime/3D-SwinUMamba-AortaSeg`
6. `cd umamba` and run `pip install -e .`
7. To set paths for nnUNet by adding folder locations to your ~/.bashrc. e.g., 
```
  export nnUNet_raw='xxx/nnUNet_raw'
  export umamba_preprocessed='xxx/nnUNet_preprocessed'
  export nnUNet_results='xxx/nnUNet_results'
```

sanity test: Enter python command-line interface and run

```bash
import torch
import mamba_ssm
```

## Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Train and inference commands
#### Train 3D models
- Train 3D `SwinUmamba3DSS3D` model

```bash
nnUNetv2_train DATASET_ID 3d_fullres all -tr nnUNetTrainerSwinUMambaRegionLoss3Dscan
```
## Inference

- Predict testing cases with `SwinUmamba3DSS3D` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerSwinUMambaRegionLoss3Dscan  
```

- Predict testing cases with `SwinUmamba3DSS3D` model

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_ID -c 3d_fullres -f all -tr nnUNetTrainerSwinUMambaRegionLoss3Dscan 
```


