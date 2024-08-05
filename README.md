# Conditional Variational AutoEncoder for Transonic Aerodynamics
[![Python](https://img.shields.io/badge/python-3.8-informational)](https://docs.python.org/3/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the implementation of a Deep Learning Framework based on Conditional Variational Autoencoder (CVAE) for the generation of a non-linear and efficient ROM for the prediction of the transonic aerodynamic simulations, but can be simply used for to the prediction of general non-linear 2D simulations. 
The description of the problem and the dataset can be found in the paper of of G.Catalani et al. [**" A comparative study of learning techniques for the compressible aerodynamics over a transonic RAE2822 airfoil"**](https://www.sciencedirect.com/science/article/abs/pii/S0045793022003516). Compared to the paper the Deep Learning framework based on a UNet is replaced with a Conditional Variational Autoencoder.

<img src="https://github.com/giovannicatalani/Aerodynamic-VAE/blob/main/images/recon_beta1_lat10.png" width="600" />

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Model](#model)
* [Dataset](#dataset)
* [Configuration](#configuration)
* [Results](#results)
* [Contributing](#contributing)



### Model
The model consists of a Convolutional Encoder, a mirrored Convolutional Decoder connected through a latent representation of the data, following the standard VAE framework. In particular, the encoder takes as input the images, in order to approximates the posterior distribution of the latent coordinates, assumed following a Gaussian distribution. The decoder reconstruct the original images from the sampled latent coordinates. The prior distribution of the latent coordinates is assumed to follow a Standard Normal distribution.
In order to be used for generation of simulation at given values of the inputs value, the posterior distribution is conditioned on the inputs value: practically both the encoder and decoder accept as additional inputs the labels (in this case [Mach Number, Angle of Attack]).
The model is trained by maximizing the Evidence Lower Bound (ELBO): the model loss consists of two terms accounting for the reconstruction loss and for the regularity of the latent space (through the KL divergence between the approximated posterior and the prior). These two terms are weighed by the paramter Beta, which can be modulted.

### Dataset
Data can be downloaded at:
https://zenodo.org/records/12700680?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQyNzI4M2NmLWIwYjktNDc1Ny1hYjA5LTliYjU4YjY4MjFmNCIsImRhdGEiOnt9LCJyYW5kb20iOiI5ZjY5MWIzNWQ5MTRmNGE4ZDdjNmY4ZjI4MTY1NDAyMiJ9._BqW0JKCMiI89PjbTmNOtbvYO6iCBx-hjP4WRPGepV2ufmAlqk_SEmAgbPfqkW9YvjOsh67lHn2jGQ7cg_n1nw

The database consist of 2D simulations over the RAE 2822 transonic airfoil at Reynolds number is set to 6.5 million, corresponding to a typical cruise flight. Two parameters have been considered in this study:


•Angle of attack: ranging from 0° to 9°.

•Mach number: ranging from 
 (incompressible) to 0.9 (transonic).

The simulations are interpolated on a pixel like grid in order to allow for convolution trhough standard convolutions.

<img src="https://github.com/giovannicatalani/Aerodynamic-VAE/blob/main/images/airfoil.jpg" width="600" />

### Configuration

The clone the repository:
```shell script
git clone git@github.com:giovannicatalani/Aerodynamic-VAE.git
```
The model uses the **PyTorch** library.
To install dependencies with conda:
```shell script
conda env create -f env_rom.yml
```

### Results

<img src="https://github.com/giovannicatalani/Aerodynamic-VAE/blob/main/images/recon_images.png" width="600" />

<img src="https://github.com/giovannicatalani/Aerodynamic-VAE/blob/main/images/lat_spaces.png" width="600" />

### Contributing
