# Extending Swin-UMamba for 3D Aortic Segmentation in CT Images with 3D-Selective-Scan Mechanism

# 3D-SwinUMamba-AortaSeg

This repository contains the implementation of **3D-SwinUMamba**, an advanced architecture for 3D aortic segmentation in CT images. The model extends the original Swin-UMamba by incorporating a novel **3D-Selective-Scan (SS3D) mechanism**, enabling efficient processing of volumetric data through six distinct scanning paths.

## Key Features:
- **3D Extension of Swin-UMamba**: Incorporates depth-wise scanning for improved volumetric segmentation in medical imaging.
- **3D-Selective-Scan (SS3D)**: Enhances long-range dependency modeling by processing patches along depth, height, and width axes in forward and reverse directions.
- **Multi-Scale Feature Extraction**: Uses a hierarchical encoder-decoder structure with feature sizes of [32, 64, 128, 256, 512] to capture both local and global anatomical details.
- **Deep Supervision**: Outputs segmentation maps for 23 distinct aortic regions, utilizing deep supervision to improve training performance.
- **Optimized for Aortic Segmentation**: Designed and tuned specifically for the segmentation of the aorta in CT images, with a focus on clinical applicability.

## Region-Based Loss Function:
This model employs a **region-based loss function** designed to enhance segmentation performance by aggregating the 23 aortic labels into four distinct anatomical regions:

- **Ascending and Arch Aorta** (labels 1–6)
- **Descending Aorta** (labels 7–9)
- **Abdominal Aorta** (labels 10–17)
- **Iliac Arteries and Peripheral Branches** (labels 18–23)

The region loss is combined with the traditional cross-entropy and dice loss applied to individual targets. This aggregation reduces label imbalance and enhances the stability of the training process.

### Total Loss Formula:
The total loss \(L_{\text{total}}\) is computed as a weighted combination of cross-entropy and dice losses, applied both to the individual target labels and the aggregated regions:

\[
L_{\text{total}} = \lambda_1 \left( L_{\text{ce-targets}} + L_{\text{dice-targets}} \right) + \lambda_2 \left( L_{\text{ce-regions}} + L_{\text{dice-regions}} \right)
\]

Where:
- \(L_{\text{ce-targets}}\) and \(L_{\text{dice-targets}}\) represent the cross-entropy and dice losses for individual target labels.
- \(L_{\text{ce-regions}}\) and \(L_{\text{dice-regions}}\) represent the same losses, applied to the aggregated regions.
- \(\lambda_1\) and \(\lambda_2\) are weighting factors to balance the contributions of target-wise and region-wise losses.

This approach promotes smoother training, balancing between different aortic regions, and improving overall segmentation accuracy.

## Getting Started
This repository includes all necessary code for training, evaluation, and reproduction of results for the AortaSeg task using the 3D-SwinUMamba architecture. It is implemented in PyTorch and optimized for 3D segmentation tasks in medical imaging.

