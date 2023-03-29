# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:48:05 2023

@author: giosp
"""

import numpy as np
import torch
from Model_CVAE import vae_loss
import matplotlib.pyplot as plt

def run_epoch(epoch, model, train_loader, optimizer,kl_weight=1, device='cpu'):
  running_loss_VAE = 0 
  running_recon_loss_VAE = 0
  running_kl_loss_VAE = 0
  
  for i, (inputs, inputs_lat ) in enumerate(train_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        inputs   = inputs.float()
        
        inputs_lat = inputs_lat.to(device)
        inputs_lat   = inputs_lat.float()

        # Zero the parameter gradients
        optimizer.zero_grad()
       
        # Forward pass
        outputs, mu, log_var = model(inputs, inputs_lat)
         # Backward pass and optimization            
        loss_VAE, recon_loss_VAE, kl_loss_VAE = vae_loss(outputs,inputs,mu,log_var,kl_weight=kl_weight)
        loss_VAE.backward()
        optimizer.step()
           
        # Print statistics
        running_loss_VAE += loss_VAE.item()
        running_recon_loss_VAE += recon_loss_VAE.item()
        running_kl_loss_VAE += kl_loss_VAE.item()
            
  return running_loss_VAE/len(train_loader), running_recon_loss_VAE/len(train_loader), running_kl_loss_VAE/len(train_loader)


def run_val(epoch, model, val_loader,kl_weight=1, device='cpu'):
  running_loss_VAE = 0
  running_recon_loss_VAE = 0
  running_kl_loss_VAE = 0

  with torch.no_grad():
     for i, (inputs, inputs_lat) in enumerate(val_loader):
        # Move the inputs and labels to the device
        inputs = inputs.to(device)
        inputs   = inputs.float()
        
        inputs_lat = inputs_lat.to(device)
        inputs_lat   = inputs_lat.float()

        # Forward pass
        outputs, mu, log_var = model(inputs, inputs_lat)
         # Backward pass and optimization            
        loss_VAE, recon_loss_VAE, kl_loss_VAE = vae_loss(outputs,inputs,mu,log_var,kl_weight=kl_weight)
                 
        # Print statistics
        running_loss_VAE += loss_VAE.item()
        running_recon_loss_VAE += recon_loss_VAE.item()
        running_kl_loss_VAE += kl_loss_VAE.item()
      
  return running_loss_VAE/len(val_loader), running_recon_loss_VAE/len(val_loader), running_kl_loss_VAE/len(val_loader)

def predict_test(model,model_path,test_loader,p_mean,p_std,control_mean,control_std,device='cpu'):
    # Load the state dict saved with torch.save
    state_dict = torch.load(model_path,map_location=device)
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict_CVAE'])
    model.eval()
       
    pred_test = np.zeros((0,128,128))
    true_test = np.zeros((0,128,128))
    control_test = np.zeros((0,2))
    
    with torch.no_grad():
        for k, (test_pressure, test_control) in enumerate(test_loader):
            
            z = torch.randn(len(test_control), model.lat_dim, device=device)

            test_control   = test_control.to(device)
            test_control   = test_control.float()
            pred = model.decode(z, test_control)
            pred[:,:,:,:] = pred[:,:,:,:]*p_std +p_mean  
            pred = pred[:,0,:,:].cpu().numpy()
            pred_test = np.concatenate((pred_test, pred),axis=0)
                       
            true = test_pressure[:,:,:,:]*p_std +p_mean            
            true = true[:,0,:,:].cpu().numpy()
            true_test = np.concatenate((true_test, true),axis=0)
            
            control = test_control.cpu().numpy()
            control_test = np.concatenate((control_test, control),axis=0)
                                    
    
    return pred_test, true_test, control_test

def generatesample(model, model_path, p_mean, p_std, num_samples, device='cpu'):
    
    # Load the state dict saved with torch.save
    state_dict = torch.load(model_path, map_location=device)
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict_VAE'])
    model.eval()
    
    with torch.no_grad():
        z = torch.randn(num_samples, model.lat_dim).to(device) # Sample from standard normal distribution
        samples = model.decode(z) # Decode the samples
    
    samples = samples * p_std + p_mean # Rescale the samples to original range
    samples = samples[:, 0, :, :].cpu().numpy() # Convert to numpy array and extract first channel
    
    return samples

def reconstruct(model,model_path,test_loader,p_mean,p_std,control_mean, control_std,device='cpu'):
    # Load the state dict saved with torch.save
    state_dict = torch.load(model_path,map_location=device)
    # Load the state dict into the model
    model.load_state_dict(state_dict['state_dict_CVAE'])
    model.eval()
    
    pred_test = np.zeros((0,128,128))
    true_test = np.zeros((0,128,128))
    control_test = np.zeros((0,2))
    mu_test = np.zeros((0,model.lat_dim))
    log_var_test = np.zeros((0,model.lat_dim))
    
    with torch.no_grad():
        for k, (test_pressure, test_control) in enumerate(test_loader):
                        
            test_pressure   = test_pressure.to(device)
            test_pressure   = test_pressure.float()

            test_control   = test_control.to(device)
            test_control   = test_control.float()
            
            pred, mu, log_var = model(test_pressure, test_control)
            pred[:,:,:,:] = pred[:,:,:,:]*p_std +p_mean  
            pred = pred[:,0,:,:].cpu().numpy()
            pred_test = np.concatenate((pred_test, pred),axis=0)
                       
            true = test_pressure[:,:,:,:]*p_std +p_mean            
            true = true[:,0,:,:].cpu().numpy()
            true_test = np.concatenate((true_test, true),axis=0)
            
            control = test_control.cpu().numpy()*control_std + control_mean
            control_test = np.concatenate((control_test, control),axis=0)
            
            mu = mu.cpu().numpy()
            mu_test = np.concatenate((mu_test, mu),axis=0)
            log_var = log_var.cpu().numpy()
            log_var_test = np.concatenate((log_var_test, log_var),axis=0)
        
    return pred_test, true_test, control_test, mu_test, log_var_test


def plot_writer(data_directory,res,P_pred,P_true,control,k,mu=None,latent_plot = False):
    
    airfoil = np.load(data_directory + 'airfoil.npy')
    #Plotting original snapshots reconstructed
    x_nodes = np.linspace(-0.25,1.25,res)
    y_nodes = np.linspace(-0.5,0.5,res)
    X, Y= np.meshgrid(x_nodes, y_nodes)
    airfoilF = np.concatenate((airfoil[:352, :], airfoil[:351:-1, :], airfoil[0:1, :]))

    plt.style.use(['science', 'ieee'])
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    #plt.contour(X,Y,P_pred[k] ,5,colors='white')
    plt.contourf(X,Y,P_pred[k], 100)
    plt.fill(airfoilF[:, 0], airfoilF[:, 1], facecolor='white', edgecolor=None)
    plt.plot(airfoilF[:, 0], airfoilF[:, 1], 'k')
    plt.title('Reconstruction')
    plt.colorbar()
        
    plt.subplot(1,2,2)
    #plt.contour(X,Y,P_true[k] ,5,colors='white')
    plt.contourf(X,Y,P_true[k], 100)
    plt.fill(airfoilF[:, 0], airfoilF[:, 1], facecolor='white', edgecolor=None)
    plt.plot(airfoilF[:, 0], airfoilF[:, 1], 'k')
    plt.title('Ground Truth')
    plt.colorbar()
    
    if latent_plot:
        from sklearn.decomposition import PCA
    
        # Perform PCA to reduce the dimensionality of the latent space to 2 dimensions
        pca = PCA(n_components=2)
        latent_space_2d = pca.fit_transform(mu)
        #Define the coloring depending on control features
    
        alpha = control[:,0]
        M = control[:,1]
        # Plot the scatter plot
        f = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
    
    
        im = ax1.scatter(mu[:, 0], mu[:, 1], c=M[:len(mu)], cmap='viridis',s = 1)
        ax1.set_xlabel('Latent dimension 1')
        ax1.set_ylabel('Latent dimension 2')
        plt.colorbar(im,ax=ax1)
    
        im = ax2.scatter(mu[:, 0], mu[:, 1], c=alpha[:len(mu)], cmap='viridis',s = 1)
        ax2.set_xlabel('Latent dimension 1')
        ax2.set_ylabel('Latent dimension 2')
        plt.colorbar(im,ax=ax2)
    
    
        # Plot the scatter plot
        f = plt.figure(figsize=(8, 3))
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ax2 = plt.subplot2grid((1, 2), (0, 1))
    
        im = ax1.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], c=M[:len(mu)], cmap='viridis',s = 1)
        ax1.set_xlabel('Latent dimension 1')
        ax1.set_ylabel('Latent dimension 2')
        plt.colorbar(im,ax=ax1)
    
        im = ax2.scatter(latent_space_2d[:, 0], latent_space_2d[:, 1], c=alpha[:len(mu)], cmap='viridis',s = 1)
        ax2.set_xlabel('Latent dimension 1')
        ax2.set_ylabel('Latent dimension 2')
        plt.colorbar(im,ax=ax2)


   