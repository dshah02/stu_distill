import sys
import os
sys.path.append(os.path.abspath("../../src"))

import torch
import torch.nn as nn
import torch.nn.functional  as  F
import math
import numpy as np

device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

from convolve import convolve, nearest_power_of_two
from stu import STU
from lds_utils import compute_ar_x_preds
from lds import LDS

flash_fft_available = False
dtype = torch.bfloat16 if flash_fft_available else torch.float32

use_hankel_L  = False
phi = torch.tensor(np.load('spectral_filters.npy')).to(device).to(dtype)
seq_len, num_eigh = 8192, 24    
# phi= get_spectral_filters(seq_len = seq_len, K = num_eigh,  use_hankel_L= use_hankel_L, device  = device,  dtype = torch.float32)
#I have checked that phi from the file and from get_spectral_filters are the same up to precision and sign
n = nearest_power_of_two(seq_len * 2 - 1, round_up=True)

class Config:
    def __init__(self):
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.torch_dtype = dtype
        self.n_embd = 1  # d_in and d_out
        self.seq_len = seq_len
        self.k_u = 0
        self.use_flash_fft = flash_fft_available
        self.use_approx = False

stu_config = Config()

import os
import torch
from pathlib import Path


def load_lds_stu_pairs(directory="../../della/lds_trained"):
    # Convert to Path object for easier path manipulation
    dir_path = Path(directory)
    
    # Dictionary to store pairs temporarily
    pairs_dict = {}
    
    # List all files in directory
    for file in dir_path.glob("*"):
        try:
            if file.suffix == '.pth':
                # Parse filename
                name = file.stem  # Get filename without extension
                
                # Handle different naming patterns
                if 'lds_' in name:
                    # Format like "99462425lds_15.pth"
                    parts = name.split('lds_')
                    prefix = parts[0]
                    number = parts[1]
                    model_type = 'lds'
                elif '_lds_' in name:
                    # Format like "prefix_lds_15.pth"
                    prefix = name.split('_lds_')[0]
                    number = name.split('_')[-1]
                    model_type = 'lds'
                elif 'stu_' in name:
                    # Format like "99462425stu_15.pth"
                    parts = name.split('stu_')
                    prefix = parts[0]
                    number = parts[1]
                    model_type = 'stu'
                elif '_stu_' in name:
                    # Format like "prefix_stu_15.pth"
                    prefix = name.split('_stu_')[0]
                    number = name.split('_')[-1]
                    model_type = 'stu'
                else:
                    # Skip files that don't match expected patterns
                    continue
                    
                # Create key for matching pairs
                key = (prefix, number)
                
                # Initialize dict entry if not exists
                if key not in pairs_dict:
                    pairs_dict[key] = {'lds': None, 'stu': None}
                
                # Load the model state dict
                model_state = torch.load(file)
                pairs_dict[key][model_type] = model_state
        except Exception as e:
            pass
    
    # Convert to list of pairs, only keeping complete pairs
    pairs = []
    for key, models in pairs_dict.items():
        if models['lds'] is not None and models['stu'] is not None:
            pairs.append((models['lds'].to(device), models['stu'].to(device)))
    
    return pairs

# Usage
lds_stu_pairs = load_lds_stu_pairs()

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
torch.manual_seed(42)
# Define a function to compute MSE between LDS and STU outputs
def compute_mse_for_pairs(pairs, seq_len=100, batch_size=1, input_dim=1):
    results = []
    
    for i, (lds, stu) in enumerate(tqdm(pairs, desc="Computing MSE")):
        # Generate Gaussian input with fixed seed for reproducibility
        
        gaussian_input = torch.randn(batch_size, seq_len, input_dim, device=device)
        
        # Set models to evaluation mode
        lds.eval()
        stu.eval()
        
        # Forward pass through both models
        with torch.no_grad():
            lds_output = lds(gaussian_input)
            stu_output = stu(gaussian_input)
            
            # Compute MSE
            mse = F.mse_loss(lds_output, stu_output).item()
            
        results.append({
            'pair_index': i,
            'mse': mse,
            'lds': lds,
            'stu': stu
        })
        
    return results

# Compute MSE for all pairs
mse_results = compute_mse_for_pairs(lds_stu_pairs, 8192)

# Filter pairs with MSE less than threshold
threshold = 4e-8
filtered_pairs = []
for result in mse_results:
    if result['mse'] < threshold:
        filtered_pairs.append((result['lds'], result['stu']))

print(f"Total pairs: {len(mse_results)}")
print(f"Filtered pairs with low MSE: {len(filtered_pairs)}")


def gen_lds_impulse(lds, seq_len = seq_len): #need the stu for to add the negative autoregressive component
    lds_impulse = torch.zeros(seq_len, device=device)
    for i in range(seq_len):
        a_power = lds.A ** i
        lds_impulse[i] += torch.sum(lds.C[:, 0] * a_power * lds.B[0])
    return lds_impulse

phi_n = phi.data.cpu().numpy()

def gen_stu_impulse(stu, seq_len = seq_len):
    alt_sign = lambda x: x * np.array([1, -1] * (seq_len//2))
    pos_coef = stu.M_phi_plus.data.cpu().numpy()[:, 0,0]
    neg_coef = stu.M_phi_minus.data.cpu().numpy()[:,0,0]
    impulse = np.sum(phi_n*pos_coef, axis = -1) + alt_sign(np.sum(phi_n*neg_coef, axis = -1))
    return impulse

# Reconstruct filters from LDS-STU pairs using pseudoinverse approach
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


print(f"Attempting to reconstruct filters from {len(filtered_pairs)} LDS-STU pairs")

# Extract LDS and STU parameters from filtered pairs
lds_params = [lds for lds, _ in filtered_pairs]
stu_params = [stu for _, stu in filtered_pairs]

# Generate impulse responses for all pairs
lds_impulses = np.array([gen_lds_impulse(lds).cpu().detach().numpy() for lds in lds_params])
stu_impulses = np.array([gen_stu_impulse(stu) for stu in stu_params])

# Convert to torch tensors
lds_impulses_tensor = torch.tensor(lds_impulses, dtype=torch.float, device=device)

# Assuming we have the combined weights from the STU parameters
# We'll construct combined_weights from STU parameters
alternating_signs = np.array([1, -1] * (seq_len//2))
phi_n_alternating = phi_n * alternating_signs[:, np.newaxis]
phi_n_combined = np.concatenate([phi_n, phi_n_alternating], axis=1)  # Shape: (1024, 40)

# For each STU, combine M_phi_plus and M_phi_minus
combined_weights = []
for stu in stu_params:
    # Get M_phi_plus and M_phi_minus from STU model
    M_phi_plus = stu.M_phi_plus.detach().cpu().numpy()[:,0,0]
    M_phi_minus = stu.M_phi_minus.detach().cpu().numpy()[:,0,0]
    
    # Concatenate the weights
    combined = np.concatenate([M_phi_plus, M_phi_minus], axis=0)
    combined_weights.append(combined)

# Stack all combined weights into a single array
combined_weights = np.stack(combined_weights)
combined_weights.shape

combined_weights_pinv = np.linalg.pinv(combined_weights.T)
phi_n_approx = np.matmul(lds_impulses.T, combined_weights_pinv)

# Convert to tensors on device for MSE calculation
phi_n_combined_tensor = torch.tensor(phi_n_combined, device=device)
phi_n_approx_tensor = torch.tensor(phi_n_approx, device=device)
mse = F.mse_loss(phi_n_combined_tensor, phi_n_approx_tensor)
print(f"MSE between original and approximated phi: {mse:.2e}")