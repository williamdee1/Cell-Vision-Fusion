<p align="center">
<img src="https://github.com/williamdee1/Cell-Vision-Fusion/blob/main/media/banner.PNG" width=75% height=75% class="center">
</p>
# Cell-Vision Fusion
## A Computer Vision-based Approach to Predict Kinase Inhibitor Mechanism of Action from Cell Painting Data

<p align="center">
<img src="https://github.com/williamdee1/Cell-Vision-Fusion/blob/main/media/fusion_overview.PNG" width=75% height=75% class="center">
</p>
  
**Abstract:** *To complete.*

biorxiv pre-print: To complete

## Primary Reference Material and Data Sources

| Path | Description
| :--- | :----------
| [JUMP Cell Painting Datasets](https://github.com/jump-cellpainting/datasets) |  JUMP GitHub Repository
| [Broad Cell Painting Galley s3 Bucket](https://s3.console.aws.amazon.com/s3/buckets/cellpainting-gallery?region=us-east-1&tab=objects#) |  Storage for Cell Painting data
| [cpg0016 Paper](https://www.biorxiv.org/content/10.1101/2023.03.23.534023v2) |  Morphological impact of 136,000 chemical and genetic perturbations



## Requirements
* 1&ndash;2 GPUs with at least 12 GB of memory.
* 64-bit Python 3.7 and PyTorch 1.8.1. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions.
* CUDA toolkit 11.0 or later. 
* Python libraries: see [reqs.txt](reqs.txt) for necessary libraries.

## Getting Started


## Training Time


| Labels      | Resolution | GPUs | 1000 kimg | 25000 kimg | sec/kimg | GPU mem | CPU mem
| :---------: | :--------: | :--: | :-------: | :--------: | :------: | :-----: | :-----:
| Binary      | 512x512    | 4    | 7h 20m    | 7d 14h     | ~25      | 8.0 GB  | 5.6 GB
| Multi-class | 512x512    | 4    | 7h 22m    | 7d 16h     | ~25      | 7.5 GB  | 5.8 GB



## Deep Learning Classifier

This repository contains the modules needed to replicate the deep learning classifier, trained to identify thyroid histopathology images as PTC-like or Non-PTC-like.

These modules must be combined with the legacy.py module, as well as the dnnlib and torch_utils folder from the [StyleGAN2-ADA repo](https://github.com/NVlabs/stylegan2-ada-pytorch).

### Training




### Evaluation
