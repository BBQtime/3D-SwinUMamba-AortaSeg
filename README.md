# Extending Swin-UMamba for 3D Aortic Segmentation in CT Images with 3D-Selective-Scan Mechanism

# 3D-SwinUMamba-AortaSeg

This repository contains the implementation of **3D-SwinUMamba**, an advanced architecture for 3D aortic segmentation in CT images. The model extends the original [UMamba]([[https://github.com/path-to-umamba-repo](https://github.com/bowang-lab/U-Mamba)](https://github.com/bowang-lab/U-Mamba)) and [Swin-UMamba]([[https://github.com/path-to-swin-umamba-repo](https://github.com/JiarunLiu/Swin-UMamba)](https://github.com/JiarunLiu/Swin-UMamba)) by incorporating a novel **3D-Selective-Scan (SS3D)** mechanism, enabling efficient processing of volumetric data through six distinct scanning paths. This repository is specifically aimed at the **AortaSeg2024 Challenge**.

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

## Getting Started
This repository includes all necessary code for training, evaluation, and reproduction of results for the AortaSeg task using the 3D-SwinUMamba architecture. It is implemented in PyTorch and optimized for 3D segmentation tasks in medical imaging.

