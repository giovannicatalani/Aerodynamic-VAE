# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:45:09 2023

@author: giosp
"""

import torch
import torch.optim as optim
from Model_CVAE import ConditionalVariationalAutoencoder
from utilities_CVAE import run_epoch, run_val, predict_test, reconstruct, plot_writer
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import DataLoader

# Directories
#data_directory = '/content/drive/MyDrive/Thesis NLR/data/'
data_directory = 'D:/Transonic_VAE/data/'
model_path = 'D:/Transonic_VAE/CVAE_ROM/models_trained/best_model_CVAE_beta20_lat10.pt'
best_model_path = 'D:/Transonic_VAE/CVAE_ROM/models_trained/Autoencoder/best_model_CVAE_beta20_lat10.pt'

#Options
train_model = False #If false pretrained best model is used for inference

# Define the device to use for training (e.g. GPU, CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the latent dimension for the autoencoder
latent_dim = 10
control_dim = 2
res= 128

# Create the models: AE and Hidden MLP
model_CVAE = ConditionalVariationalAutoencoder(latent_dim, control_dim, res)

# Move the models to the device
model_CVAE = model_CVAE.to(device)

# Define a loss function and optimizer
optimizer = optim.Adam(model_CVAE.parameters(), lr=0.0001, weight_decay=1e-5)
batch_size = 32

#KL weight
kl_weight = 1.0

# Import Data and Create Dataset
train_loader, valid_loader, test_loader, p_mean, p_std, control_mean, control_std = DataLoader(
    data_directory, res, batch_size)

print('Dataset creation: done \n')

#%% Training
if train_model:
    state = {}
    state['best_valid_loss'] = float('inf')
    aux_loss = np.empty([6, 1000])  # store loss values
    print('Starting training \n')
    for epoch in range(1000):
        tr_loss, tr_loss_recon, tr_loss_kl = run_epoch(
            epoch, model_CVAE, train_loader,  optimizer,kl_weight=kl_weight, device=device)
        valid_loss, valid_loss_recon, valid_loss_kl = run_val(
            epoch, model_CVAE,  valid_loader, kl_weight=kl_weight, device=device)
        aux_loss[0, epoch] = tr_loss
        aux_loss[1, epoch] = tr_loss_recon
        aux_loss[2, epoch] = tr_loss_kl
        aux_loss[3, epoch] = valid_loss
        aux_loss[4, epoch] = valid_loss_recon
        aux_loss[5, epoch] = valid_loss_kl
        
        if valid_loss < state['best_valid_loss']:
            state['best_valid_loss'] = valid_loss
            state['epoch'] = epoch
            state['state_dict_CVAE'] = model_CVAE.state_dict()
            state['optimizer_CVAE'] = optimizer.state_dict()

            torch.save(state, model_path)

        print('Train loss CVAE: {:.6f} \n'.format(tr_loss))
        print('Train loss Reconstruction: {:.6f} \n'.format(tr_loss_recon))
        print('Train loss KL: {:.6f} \n'.format(tr_loss_kl))

        print('Valid loss CVAE: {:.6f} \n'.format(valid_loss))
        print('Valid loss Reconstruction: {:.6f} \n'.format(valid_loss_recon))
        print('Valid loss KL: {:.6f} \n'.format(valid_loss_kl))

        plt.figure()
        plt.plot(np.arange(epoch), aux_loss[0, :epoch], label='training CVAE')
        plt.plot(np.arange(epoch), aux_loss[1, :epoch], label='training reconstruction')
        plt.plot(np.arange(epoch), aux_loss[2, :epoch], label='training KL')
        plt.plot(np.arange(epoch), aux_loss[3, :epoch], label='validation CVAE')
        plt.plot(np.arange(epoch), aux_loss[4, :epoch], label='validation reconstruction')
        plt.plot(np.arange(epoch),
                aux_loss[5, :epoch], label='validation KL')
        plt.legend()
        plt.yscale('log')
        plt.show()

    print('Finished Training')

#%% Reconstruction of pressure fields on train and latent vecotors set
pred_train, true_train, control_train, mu_train, log_var_train = reconstruct(model_CVAE,model_path,train_loader,p_mean,p_std,control_mean,control_std,device='cpu')

#%% Plotting Reconstruction and Latent Space Visualization
k=21 #Idx of snapshot to plot
plot_writer(data_directory,res,pred_train,true_train,control_train,k,mu=mu_train,latent_plot=True)   

#%%Prediction on test set
pred_test, true_test, control_test = predict_test(model_CVAE,model_path,test_loader,p_mean,p_std,control_mean,control_std,device='cpu')
#%% Plotting Predictions
k=44 #Idx of snapshot to plot
plot_writer(data_directory,res,pred_test,true_test,control_test,k)  
