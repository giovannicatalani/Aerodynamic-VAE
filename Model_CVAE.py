# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:09:11 2023

@author: giosp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self,lat_dim, y_dim, img_res):
        """
        Initializes the Conditional Variational Autoencoder module.

        Parameters:
        lat_dim (int): Size of the latent vector (e.g. 5, 10, 20)
        y_dim (int): Size of the control vector
        img_res (int): Resolution of the input images (assumed to be square)
        """
        super(ConditionalVariationalAutoencoder, self).__init__()

        # Encoder
        self.img_res = img_res
        self.lat_dim = lat_dim
        self.y_dim = y_dim

        self.encoder = nn.Sequential(
            # Add your encoder layers here
            # First Conv Block
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), # out [64,128,128]
            nn.LeakyReLU(negative_slope=0.25),
            nn.MaxPool2d(kernel_size=2, stride=2), # out [64,64,64]
            # Second Conv Block
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1), # out [32,64,64]
            nn.LeakyReLU(negative_slope=0.25),
            nn.MaxPool2d(kernel_size=2, stride=2), # out [32,32,32]
            # Third Conv Block
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1), # out [16,32,32]
            nn.LeakyReLU(negative_slope=0.25),
            nn.MaxPool2d(kernel_size=2, stride=2), # out [16,16,16]
            # Flatten and Dense Layer to latent space
            nn.Flatten()
        )

        self.conditional_encoder = nn.Sequential(
            nn.Linear(in_features=16*(self.img_res // 8)*(self.img_res // 8) + self.y_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.25)
        )

        self.fc_mu = nn.Linear(128, lat_dim)
        self.fc_logvar = nn.Linear(128, lat_dim)

        self.intermediate = nn.Sequential(
            nn.Linear(in_features=lat_dim + self.y_dim, out_features=128),
            nn.LeakyReLU(negative_slope=0.25),
            nn.Linear(in_features=128, out_features=16*(self.img_res // 8)*(self.img_res // 8)),
            nn.LeakyReLU(negative_slope=0.25)
        )

        self.decoder = nn.Sequential(
            # Add your decoder layers here
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.25),
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.25),
            # Upsample the feature maps and pass them through a convolutional layer
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(negative_slope=0.25),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=1),
        )

    def encode(self, x, y):
        """
        Encodes the input images and control vector into
        Parameters:
        x (torch.Tensor): Input images
        y (torch.Tensor): Control vector

        Returns:
        mu (torch.Tensor): Mean of the latent space
        logvar (torch.Tensor): Log variance of the latent space
        """
        x = self.encoder(x)
        xy = torch.cat([x, y], dim=1)
        xy = self.conditional_encoder(xy)
        mu = self.fc_mu(xy)
        logvar = self.fc_logvar(xy)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterizes the latent space using the mean and log variance.

        Parameters:
        mu (torch.Tensor): Mean of the latent space
        logvar (torch.Tensor): Log variance of the latent space

        Returns:
        z (torch.Tensor): Reparameterized latent space
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z, y):
        """
        Decodes the reparameterized latent space and control vector into
        reconstructed images.

        Parameters:
        z (torch.Tensor): Reparameterized latent space
        y (torch.Tensor): Control vector

        Returns:
        recon_x (torch.Tensor): Reconstructed images
        """
        zy = torch.cat([z, y], dim=1)
        zy = self.intermediate(zy)
        zy = zy.reshape(zy.shape[0], -1, self.img_res // 8, self.img_res // 8)
        recon_x = self.decoder(zy)
        return recon_x

    def forward(self, x, y):
        """
        Computes the forward pass of the VAE given the input images and control vector.

        Parameters:
        x (torch.Tensor): Input images
        y (torch.Tensor): Control vector

        Returns:
        recon_x (torch.Tensor): Reconstructed images
        mu (torch.Tensor): Mean of the latent space
        logvar (torch.Tensor): Log variance of the latent space
        """
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, y)
        return recon_x, mu, logvar

    def sample(self, num_samples):
        """
        Generates samples from the VAE by sampling from the prior distribution
        and decoding the samples.

        Parameters:
        num_samples (int): Number of samples to generate

        Returns:
        samples (torch.Tensor): Generated samples
        """
        z = torch.randn((num_samples, self.lat_dim)).to(next(self.parameters()).device)
        y = torch.zeros((num_samples, self.y_dim)).to(next(self.parameters()).device)
        samples = self.decode(z, y)
        return samples

def vae_loss(recon_x, x, mu, logvar, kl_weight = 1.0):
    """
    Computes the VAE loss given the reconstructed images, the original images,
    the mean and log variance of the latent space, and the KL weight.

    Parameters:
    recon_x (torch.Tensor): Reconstructed images
    x (torch.Tensor): Original images
    mu (torch.Tensor): Mean of the latent space
    logvar (torch.Tensor): Log variance of the latent space
    kl_weight (float): Weight for the KL divergence term

    Returns:
    loss (torch.Tensor): Total VAE loss
    mse_loss (torch.Tensor): Reconstruction loss
    kld_loss (torch.Tensor): KL divergence loss
    """
    # Reconstruction loss
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence loss
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Total VAE loss
    loss = mse_loss + kl_weight * kld_loss
    return loss, mse_loss, kld_loss