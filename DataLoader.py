# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:14:36 2023

@author: giosp
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split



def load_data(data_directory, file, res):
    
    db = np.load(data_directory + file, allow_pickle=True).item()
   
    pressure = np.float32(db['P_' + str(res) + 'x' + str(res)])
    controls = np.zeros((len(db['Alpha']),2))

    controls[:,0] = db['Alpha']
    controls[:,1] = db['Vinf']/347
    
    return pressure, controls

def compute_mean_std(data):
    return data.mean((0)), data.std((0))

def normalize(data, mean, std):
    return (data - mean) / std

def DataLoader(data_directory, res, batch_size,shuffle=True):
    
    # Load data (Dict)
    file_cyc = 'db_cyc_' + str(res) + '.npy'
    file_random = 'db_random_' + str(res) + '.npy'
    
    pressure_cyc, controls_cyc = load_data(data_directory, file_cyc, res)
    pressure_random, controls_random = load_data(data_directory, file_random, res)
    
    pressure = np.concatenate((pressure_cyc, pressure_random), axis=0)
    controls = np.concatenate((controls_cyc, controls_random), axis=0)
       
    # Compute Mean and Std Pressure
    pressure_mean, pressure_std = compute_mean_std(pressure)

    # Compute Mean and Std Controls
    control_mean, control_std = compute_mean_std(controls)

    # Normalize
    pressure = normalize(pressure, pressure_mean, pressure_std)
    controls = normalize(controls, control_mean, control_std)
    
    train_pressure, valid_pressure, train_controls, valid_controls = train_test_split(pressure, controls, test_size=0.2, random_state=42)
    test_pressure, valid_pressure, test_controls, valid_controls = train_test_split(valid_pressure, valid_controls, test_size=0.5, random_state=42)
    
    # Convert to Tensor
    train_pressure = torch.from_numpy(train_pressure.astype(np.float))
    train_controls = torch.from_numpy(train_controls.astype(np.float))
    valid_pressure = torch.from_numpy(valid_pressure.astype(np.float))
    valid_controls = torch.from_numpy(valid_controls.astype(np.float))
    test_pressure = torch.from_numpy(test_pressure.astype(np.float))
    test_controls = torch.from_numpy(test_controls.astype(np.float))

    # Add dim to pressure to be processed by AE
    train_pressure = train_pressure.unsqueeze(1)
    valid_pressure = valid_pressure.unsqueeze(1)
    test_pressure = test_pressure.unsqueeze(1)
      
   
    # Define DataLoaders
    train_ds = torch.utils.data.TensorDataset(train_pressure, train_controls)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)

    valid_ds = torch.utils.data.TensorDataset(valid_pressure, valid_controls)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=shuffle)
    
    test_ds = torch.utils.data.TensorDataset(test_pressure, test_controls)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader, test_loader, pressure_mean, pressure_std, control_mean, control_std
