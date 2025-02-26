import argparse

parser = argparse.ArgumentParser(description='Run the script with a sequence length.')
parser.add_argument('--seq_len', type=int, required=True, help='Sequence length')
args = parser.parse_args()
seq_len = args.seq_len

import sys
import os
sys.path.append(os.path.abspath("../src"))

try:
    from flashfftconv import FlashFFTConv

    flash_fft_available = True
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False


import argparse
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from model_550m import STU, flash_convolve
import time
import random
from torch.nn import functional as F

from lds import LDS

layer_i = 2
state_dim = 1000
batch_size = 2
epochs = 4000
seq_len = 8192
kx = 5
lr = 0.0001

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the layer i weights
stu_layer_full = torch.load(f"../stu_layers/stu_layer_{layer_i}_550m_param_full.pt", map_location=device)
stu_layer_full.eval()

# Initialize LDS model
lds = LDS(state_dim, 896, 896, kx).to(device)
optimizer = torch.optim.Adam(lds.parameters(), lr=lr)

# Training
lds_loss_values = []

best_loss = float('inf')

phi = stu_layer_full.stu_filters
seq_len = phi.shape[0]

def gen_stu_impulse_approx(stu, seq_len=1000):
    """
    Generate the impulse response of a STU model with approximation.
    
    Args:
        stu: The STU model
        seq_len: Length of the impulse response
        
    Returns:
        impulse_response: The impulse response of the STU model with shape (seq_len, d_out, d_in)
    """
    # Create an impulse input
    batch_size = 1
    d_in = stu.d_in
    d_out = stu.d_out
    impulse = torch.zeros((batch_size, seq_len, d_in), device=stu.M_inputs.device if hasattr(stu, 'M_inputs') else 'cpu')
    
    # Initialize the output tensor with the correct shape (seq_len, d_out, d_in)
    impulse_response = torch.zeros((seq_len, d_out, d_in), device=impulse.device)
    
    # For each input dimension, create an impulse and get the response
    for i in range(d_in):
        # Reset the impulse tensor
        impulse.zero_()
        # Set the impulse for the current input dimension
        impulse[:, 0, i] = 1.0
        
        # Pass the impulse through the STU model
        with torch.no_grad():
            if stu.use_approx:
                # Project the impulse using M_inputs
                impulse_proj = impulse @ stu.M_inputs.float()
                
                # Project the filters using M_filters
                phi_proj = stu.stu_filters.float() @ stu.M_filters.float()
                
                # Compute the convolution
                if stu.flash_fft:
                    spectral_plus, spectral_minus = flash_convolve(
                        impulse_proj, phi_proj, stu.flash_fft, stu.use_approx
                    )
                else:
                    spectral_plus, spectral_minus = convolve(
                        impulse_proj, phi_proj, stu.n, stu.use_approx
                    )
                
                # The impulse response for this input dimension
                response = spectral_plus if stu.use_hankel_L else spectral_plus + spectral_minus
            else:
                # For non-approximation case, use the original forward pass
                response = stu(impulse)
            
            # Store the response for this input dimension
            impulse_response[:, :, i] = response.squeeze(0).float()
    
    return impulse_response.cpu().numpy()


stu_impulse = gen_stu_impulse_approx(stu_layer_full, seq_len = 512)
stu_impulse = torch.Tensor(stu_impulse).cuda()

stu_impulse.shape

for epoch in range(epochs):
    # Generate impulse response from STU
    
    
    # Generate impulse response from LDS
    lds_impulse = lds.impulse(seq_len=512)

    optimizer.zero_grad()
    # Compare impulse responses
    loss = F.mse_loss(lds_impulse, stu_impulse)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lds.parameters(), max_norm=1)
    lds_loss_values.append(loss.item())
    optimizer.step()

    with torch.no_grad():
        lds.A.data.clamp_(max=1, min=-1)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(lds.state_dict(), "lds_10k_10_impulse.pth")



