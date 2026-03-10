# Conditional Variational AutoEncoder for Transonic Aerodynamics
[![Python](https://img.shields.io/badge/python-3.8-informational)](https://docs.python.org/3/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the implementation of a Deep Learning Framework based on Conditional Variational Autoencoder (CVAE) for the generation of a non-linear and efficient ROM for the prediction of the transonic aerodynamic simulations, but can be simply used for to the prediction of general non-linear 2D simulations. 
The description of the problem and the dataset can be found in the paper of of G.Catalani et al. [**" A comparative study of learning techniques for the compressible aerodynamics over a transonic RAE2822 airfoil"**](https://www.sciencedirect.com/science/article/abs/pii/S0045793022003516). Compared to the paper the Deep Learning framework based on a UNet is replaced with a Conditional Variational Autoencoder.

<img src="https://github.com/giovannicatalani/Aerodynamic-VAE/blob/main/images/recon_beta1_lat10.png" width="600" />

# Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Results](#results)

---

## Project Structure

```
.
├── data/                        # Dataset files (downloaded separately)
│   ├── db_cyc.npy               # Clenshaw-Curtis sampled simulations (raw)
│   ├── db_random.npy            # Randomly sampled simulations (raw)
│   ├── db_cyc_128.npy           # Preprocessed 128x128 grid (generated)
│   ├── db_random_128.npy        # Preprocessed 128x128 grid (generated)
│   └── airfoil.npy              # RAE2822 airfoil coordinates (for plotting)
│
└── CVAE_ROM/                    # Code
    ├── preprocess.py            # Interpolates raw data to pixel grid
    ├── DataLoader.py            # PyTorch dataset and dataloaders
    ├── Model_CVAE.py            # CVAE model definition and loss
    ├── utilities_CVAE.py        # Training, validation, inference helpers
    ├── main_CVAE.py             # Main entry point (train / infer)
    └── README.md
```

---

## Setup

Clone the repository and install dependencies with conda:

```bash
git clone https://github.com/giovannicatalani/Aerodynamic-VAE.git
cd Aerodynamic-VAE
conda env create -f env_rom.yml
conda activate rom
```

---

## Dataset

Download the dataset from Zenodo:

**[https://zenodo.org/records/12700680](https://zenodo.org/records/12700680)**

Extract `data.zip` so that the `data/` folder sits **one level above** the `CVAE_ROM/` code folder:

```
Transonic_VAE/
├── data/          <-- extracted here
└── CVAE_ROM/      <-- code here
```

The raw dataset contains unstructured CFD simulation results for the RAE2822 airfoil:

| Parameter | Range |
|-----------|-------|
| Mach number (`Vinf`) | 0.0 – 0.9 |
| Angle of attack (`Alpha`) | 0° – 9° |
| Samples per file | ~1000 |
| Nodes per simulation | 27,499 (unstructured mesh) |

---

## Preprocessing

The raw dataset uses an **unstructured CFD mesh** (scattered points). The CVAE requires a regular pixel grid for convolutions. Run the preprocessing script once to interpolate each simulation onto a `128×128` grid:

```bash
cd CVAE_ROM
python preprocess.py
```

This reads `../data/db_cyc.npy` and `../data/db_random.npy`, and writes `../data/db_cyc_128.npy` and `../data/db_random_128.npy`. Expect it to take a few minutes (scipy `griddata` per sample).

To generate a different resolution (e.g. 128):

```bash
python preprocess.py --res 128
```

---

## Training

```bash
python main_CVAE.py --train
```

Key options:

```
--train                 Run training (default: inference only)
--data_dir DIR          Path to data folder (default: ../data/)
--model_path PATH       Where to save/load the model checkpoint
--res INT               Grid resolution (default: 128)
--latent_dim INT        Latent space dimension (default: 10)
--epochs INT            Number of training epochs (default: 1000)
--batch_size INT        Batch size (default: 32)
--lr FLOAT              Learning rate (default: 1e-4)
--kl_weight FLOAT       Weight for KL divergence term / Beta (default: 1.0)
```

Example — train with a higher beta and larger latent space:

```bash
python main_CVAE.py --train --latent_dim 20 --kl_weight 20 --epochs 500
```

The best model (lowest validation loss) is saved automatically to `--model_path`.

---

## Inference

By default (no `--train` flag), the script loads the saved checkpoint and runs reconstruction on the training set and prediction on the test set, then plots results:

```bash
python main_CVAE.py
```

To point to a specific saved model:

```bash
python main_CVAE.py --model_path ../models/my_model.pt
```

---

## Model Architecture

The CVAE conditions both the encoder and decoder on the input flow parameters (Mach number, angle of attack):

```
Encoder:  Image (1×128×128)
          → Conv blocks (64→32→16 channels, MaxPool ×3)
          → Flatten → Concat with controls
          → FC(128) → μ, log σ²  (latent_dim)

Decoder:  z + controls
          → FC(128) → FC(16×16×16)
          → Reshape → ConvTranspose blocks (×3)
          → Output (1×128×128)
```

The loss is the standard β-VAE ELBO:

```
Loss = MSE(reconstruction, input) + β × KL(q(z|x,y) || p(z))
```

A higher `β` (`--kl_weight`) encourages a more structured, disentangled latent space at the cost of reconstruction sharpness.

---

## Results

<img src="https://github.com/giovannicatalani/Aerodynamic-VAE/blob/main/images/recon_images.png" width="600" />

<img src="https://github.com/giovannicatalani/Aerodynamic-VAE/blob/main/images/lat_spaces.png" width="600" />



