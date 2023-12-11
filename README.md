## Uncertainty Quantification via Neural Posterior Principal Components (NPPC)

Offical paper repository

<a href="https://arxiv.org/abs/2309.15533">Arxiv</a>

This repository contains examples for training the following datasets and distortions:

- MNIST + Inpainting
- MNIST + Denosing
- CelebA HQ + Inpainting eyes
- (more coming)

The *nppc* folder contains the main code, while the *run_\*.py* file contains scripts for running the examples. As NPPC requires a trained restoration model, each example has both a *run_\*_restoration.py* file and a *run_\*_nppc.py* file.

The *minimal_example.ipynb* file is a self contained notebook and can be viewed in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EliasNehme/NPPC/blob/main/minimal_example.ipynb)

## Training a restoration model

Training a restoration model can be done using the following code (see the script files for examples):

```python
import nppc

model = nppc.RestorationModel(
    dataset=...,          # Options: "mnist" or "celeba_hq_256". A string selecting the dataset. Can be either 
    data_folder=...,      # The path to the folder containing the dataset (the folder above the dataset's folder).
    distortion_type=...,  # Options: "inpainting_1", "inpainting_2", "denoising_1", "colorization_1" or "super_resolution_1".
                          # A string selecting the distortion type.
    net_type=...,         # Options: "unet", "res_unet" or "res_cnn". A string selecting the type of network to be used.
    lr=...,               # The learning rate.
    device=...,           # A string selecting the device to be used (i.e. *cpu*, *cuda:0*, etc.).
)
trainer = nppc.RestorationTrainer(
    model=model,          #
    batch_size=...,       # The batch size.
    output_folder=...,    # The folder in which the result should be stored.
)
trainer.train(
    n_steps=...,          # The amount of update steps to perform during training.
)
```

The optional distortions are:

- *inpainting_1*: Erasing the top 20 rows of the image (has been used on MNIST in the paper).
- *inpainting_2*: Erasing the area around the eyes in CelebA 256x256 images.
- *denoising_1*: Adding noise with STD=1.
- *colorization_1*: Turning a color image into a grayscale image.
- *super_resolution_1*: Downscaling an image by 4x.

The optional networks are:

- *unet*: A small U-Net architecture (suited for the MNIST images).
- *res_unet*: A U-Net using residual blocks (similar to the networks used in DDPM).
- *res_cnn*: A network constructed out of a series of residual blocks (similar to EDSR).

## Training an NPPC model

Training an NPPC model can be done using the following code (see the script files for examples):

```python
import nppc

model = nppc.NPPCModel(
    restoration_model_folder=...,  # The path to the folder containing the trained restoration model.
    net_type=...,                  # Options: "unet", "res_unet" or "res_cnn". A string selecting the type of network to be used.
    n_dirs=...,                    # The number of PCs to predict.
    lr=...,                        # The learning rate.
    device=...,                    # A string selecting the device to be used (i.e. *cpu*, *cuda:0*, etc.).
)

## Train
## -----
trainer = nppc.NPPCTrainer(
    model=model,
    batch_size=...,     # The batch size.
    output_folder=...,  # The folder in which the result should be stored.
)
trainer.train(
    n_steps=...,        # The amount of update steps to perform during training.
)
```
