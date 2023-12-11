## Uncertainty Quantification via Neural Posterior Principal Components (NPPC)

Offical paper repository

<a href="https://arxiv.org/abs/2309.15533">Arxiv</a>

This repository contains examples for training the following datasets and distortions:

- MNIST + Inpainting
- MNIST + Denosing
- CelebA HQ (coming soon in the following days)

The *nppc* folder contains the main code, while the *run_\*.py* file contains scripts for running the examples. As NPPC requires a trained restoration model, each example has both a *run_\*_restoration.py* file and a *run_\*_nppc.py* file.

## Training a restoration model

Training a restoration model can be done using the following code (see the script files for examples):

```python
import nppc

model = nppc.RestorationModel(
    dataset=...,
    data_folder=...,
    distortion_type=...,
    net_type=...,
    lr=...,
    device=...,
)
trainer = nppc.RestorationTrainer(
    model=model,
    batch_size=...,
    output_folder=...,
)
trainer.train(
    n_steps=...,
)
```

Where the dots should be replaced according to:

- *dataset*: a string selecting the dataset. Can be either "mnist" or "celeba_hq_256"
- *data_folder*: the path to the folder containing the dataset (the folder above the dataset's folder).
- *distortion type*: a string selecting the distortion type. Can be either:
  - *inpainting_1*: Erasing the top 20 rows of the image (has been used on MNIST in the paper).
  - *inpainting_2*: Erasing the area around the eyes in CelebA 256x256 images.
  - *denoising_1*: Adding noise with STD=1.
  - *colorization_1*: Turning a color image into a grayscale image.
  - *super_resolution_1*: Downscaling an image by 4x.
- *distortion type*: a string selecting the type of network to use. Can be either:
  - *unet*: A small U-Net architecture (suited for the MNIST images).
  - *res_unet*: A U-Net using residual blocks (similar to the networks used in DDPM).
  - *res_cnn*: A network constructed out of a series of residual blocks (similar to EDSR).
- *lr*: the learning rate.
- *device*: a string selecting the device to use (i.e. *cpu*, *cuda:0*, etc.).
- *batch_size*: the batch size.
- *output_folder*: The folder in which the result should be stored.
- *n_steps*: The amount of update steps to perform during training.

## Training an NPPC model

Training an NPPC model can be done using the following code (see the script files for examples):

```python
import nppc

model = nppc.NPPCModel(
    restoration_model_folder=...,
    net_type=...,
    n_dirs=...,
    lr=...,
    device=...,
)

## Train
## -----
trainer = nppc.NPPCTrainer(
    model=model,
    batch_size=...,
    output_folder=...,
)
trainer.train(
    n_steps=...,
)
```

With the following additional fields:

- *restoration_model_folder*: the path to the folder containing the trained restoration model.
- *n_dirs*: The number of PCs to predict.
