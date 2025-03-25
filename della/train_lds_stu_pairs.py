# In this notebook, I also attempt to find how each STU filter can be represented as a linear combination of LDSs
# However, to do this we will fit many STUs to LDSs and then use the learned weights to represent the STU as a linear combination of LDSs

import sys
import os
import argparse
sys.path.append(os.path.abspath("../src"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import tqdm
import random
# Parse command line arguments

uid = str(random.random())[2:10]

parser = argparse.ArgumentParser(description='Train STU models on random LDS systems')
parser.add_argument('--steps', type=int, default=5000, help='Number of training steps for each STU')
parser.add_argument('--num_models', type=int, default=200, help='Number of LDS-STU pairs to train')
parser.add_argument('--prefix', type=str, default=uid, help='Prefix for saved model filenames')
parser.add_argument('--batch', type=int, default=1, help='batch size')
parser.add_argument('--lb', type=float, default=0.95, help='lds lambda lower_bound')
args = parser.parse_args()

try:
    from flashfftconv import FlashFFTConv
    
    flash_fft_available = True
    if not torch.cuda.is_available():
        flash_fft_available = False
except ImportError as e:
    print(
        f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation."
    )
    flash_fft_available = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import necessary modules
from convolve import convolve, nearest_power_of_two
from stu import STU
from lds_utils import compute_ar_x_preds
from lds import LDS

def random_LDS(d_h: int, d_o: int, d_u: int, lower_bound: float):
  """
  makes a random LDS with hidden state dimension d_h, observation dimension d_o, and control dimension d_u.
  `lower_bound` is a float in [0, 1] specifying the minimum absolute value for entries in A.
  Each entry in A will be in [lower_bound, 1] multiplied by +/-1 with equal probability.
  """
  # Create LDS instance
  lds = LDS(state_dim=d_h, input_dim=d_u, output_dim=d_o, kx=0, dtype=torch.float32)
  
  # Override the A parameter with custom initialization
  A_values = torch.rand(d_h) * (1 - lower_bound) + lower_bound
  signs = torch.randint(0, 2, (d_h,)) * 2 - 1
  lds.A = nn.Parameter((A_values * signs.float()).to(device))
  
  # Initialize other parameters randomly
  lds.B = nn.Parameter(torch.randn(d_u, d_h).to(device) / d_u)
  lds.C = nn.Parameter(torch.randn(d_h, d_o).to(device) / d_h)
  lds.h0 = nn.Parameter(torch.zeros(d_h).to(device))
  
  return lds

d_h = 1
d_in = 1
d_out = 1

dtype = torch.bfloat16 if flash_fft_available else torch.float32

use_hankel_L  = False
phi = torch.tensor(np.load('spectral_filters.npy')).to(device).to(dtype)
seq_len, num_eigh = 8192, 24
# phi= get_spectral_filters(seq_len = seq_len, K = num_eigh,  use_hankel_L= use_hankel_L, device  = device,  dtype = torch.float32)
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

def train_stu(lds, steps, verbose=True):
    model = STU(stu_config, phi, n).to(device)
    lr = 1
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    model.train()

    bsz = args.batch
    steps = min(args.steps, 3 * steps//bsz)

    for step in range(steps):
        inputs = torch.randn(bsz * seq_len, d_in).to(device)
        
        # Use torch.no_grad() to avoid storing gradient information for LDS
        with torch.no_grad():
            targets = lds.generate_trajectory(inputs)

        inputs = inputs.reshape(bsz, seq_len, d_in).to(device).type(dtype)
        targets = targets.reshape(bsz, seq_len, d_out).to(device).type(dtype)
        outputs = model.forward(inputs)
        # print(outputs, targets)
        loss = F.mse_loss(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model, loss

# stu_model, _ = train_stu(lds)

# Create directory if it doesn't exist
os.makedirs('lds_trained_2', exist_ok=True)

for i in tqdm.tqdm(range(args.num_models)):
    new_lds = random_LDS(d_h=d_h, d_o=d_out, d_u=d_in, lower_bound=args.lb)
    
    # Fit STU to the LDS
    stu, loss = train_stu(new_lds, steps=args.steps, verbose=False)
    
    # # Extract positive and negative weights
    # pos_weights = stu.M_phi_plus[:, 0, 0].detach().cpu().numpy()  # Shape: [num_filters]
    # neg_weights = stu.M_phi_minus[:, 0, 0].detach().cpu().numpy()  # Shape: [num_filters]
    # auto_reg = stu.M.detach().cpu().numpy()
    
    # Save the models after moving to CPU
    lds_cpu = new_lds.cpu()
    stu_cpu = stu.cpu()
    
    # Save models with the specified prefix
    torch.save(lds_cpu, f'lds_trained/{args.prefix}lds_{i}.pth')
    torch.save(stu_cpu, f'lds_trained/{args.prefix}stu_{i}.pth')
    print(loss)
