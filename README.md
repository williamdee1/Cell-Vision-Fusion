<p align="center">
<img src="https://github.com/williamdee1/Cell-Vision-Fusion/blob/main/media/banner.PNG" width=100% height=100% class="center">
</p>

**Abstract:** *Image-based profiling of the cellular response to drug compounds has proven to be an effective method to characterize the morphological changes resulting from chemical perturbation experiments. This approach has been useful in the field of drug discovery, ranging from phenotype-based screening to identifying a compound's mechanism of action or toxicity. As a greater amount of data becomes available however, there are growing demands for deep learning methods to be applied to perturbation data. In this paper we applied the transformer-based SwinV2 computer vision architecture to predict the mechanism of action of ten kinase inhibitor compounds directly from raw images of the cellular response. This method outperforms the standard approach of using image-based profiles, multidimensional feature set representations generated by bioimaging software. Furthermore, we combined the best-performing models for three different data modalities, raw images, image-based profiles and compound chemical structures, to form a fusion model, Cell-Vision Fusion (CVF). This approach classified the kinase inhibitors with 69.79\% accuracy and 70.56\% F1 score, 4.2\% and 5.49\% greater, respectively, than the best performing image-based profile method. Our work provides three techniques, specific to Cell Painting images, which enable the SwinV2 architecture to train effectively, as well as exploring approaches to combat the significant batch effects present in large Cell Painting perturbation datasets.*

biorxiv pre-print: To complete

## Approach Overview
<p align="center">
<img src="https://github.com/williamdee1/Cell-Vision-Fusion/blob/main/media/fusion_overview.PNG" width=75% height=75% class="center">
</p>
  


## Primary Reference Material and Data Sources

| Path | Description
| :--- | :----------
| [JUMP Cell Painting Repository](https://github.com/jump-cellpainting/datasets) |  JUMP Consortium GitHub Repository
| [Broad Cell Painting Galley s3 Bucket](https://s3.console.aws.amazon.com/s3/buckets/cellpainting-gallery?region=us-east-1&tab=objects#) |  AWS S3 Storage bucket for Cell Painting data
| [JUMP Cpg0016 Paper](https://www.biorxiv.org/content/10.1101/2023.03.23.534023v2) |  Morphological impact of 136,000 chemical and genetic perturbations paper - Chandrasekaran et al.



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
