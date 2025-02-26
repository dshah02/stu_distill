#!/usr/bin/env python3
import sys
import os
import argparse
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from torch.nn import functional as F

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train LDS to approximate STU impulse responses")
parser.add_argument("--layer_i", type=int, default=2, help="Layer index to load")
parser.add_argument("--state_dim", type=int, default=1000, help="Dimension of LDS state")
parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
parser.add_argument("--kx", type=int, default=5, help="KX parameter for LDS")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs")
parser.add_argument("--output_dir", type=str, default="lds_results", help="Output directory")
parser.add_argument("--stu_layer_path", type=str, default="../stu_layers", 
                    help="Path to directory containing STU layers")
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Add src directory to path for imports
sys.path.append(os.path.abspath("../src"))

# Try to import FlashFFTConv
try:
    from flashfftconv import FlashFFTConv
    flash_fft_available = True
except ImportError as e:
    print(f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation.")
    flash_fft_available = False

# Import the LDS model
from lds import LDS

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Function to generate STU impulse approximation
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
                    from model_550m import flash_convolve
                    spectral_plus, spectral_minus = flash_convolve(
                        impulse_proj, phi_proj, stu.flash_fft, stu.use_approx
                    )
                else:
                    from model_550m import convolve
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

def main():
    # Load parameters from args
    layer_i = args.layer_i
    state_dim = args.state_dim
    seq_len = args.seq_len
    kx = args.kx
    lr = args.lr
    epochs = args.epochs
    
    # Load the layer i weights
    stu_layer_path = f"{args.stu_layer_path}/stu_layer_{layer_i}_550m_param_full.pt"
    print(f"Loading STU layer from: {stu_layer_path}")
    stu_layer_full = torch.load(stu_layer_path, map_location=device)
    stu_layer_full.eval()
    
    # Get STU impulse response
    print("Generating STU impulse response...")
    stu_impulse = gen_stu_impulse_approx(stu_layer_full, seq_len=seq_len)
    stu_impulse = torch.Tensor(stu_impulse).to(device)
    
    # Save STU impulse response and filters
    torch.save(stu_impulse.cpu(), f"{args.output_dir}/filter_{layer_i}_impulse.pth")
    torch.save(stu_layer_full.stu_filters.cpu(), f"{args.output_dir}/phi.pth")
    
    print(f"STU impulse shape: {stu_impulse.shape}")
    
    # Initialize LDS model
    input_dim = stu_impulse.shape[2]
    output_dim = stu_impulse.shape[1]
    print(f"Initializing LDS with state_dim={state_dim}, input_dim={input_dim}, output_dim={output_dim}, kx={kx}")
    lds = LDS(state_dim, input_dim, output_dim, kx).to(device)
    optimizer = torch.optim.Adam(lds.parameters(), lr=lr)
    
    # Training
    print(f"Starting training for {epochs} epochs...")
    lds_loss_values = []
    best_loss = float('inf')
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        running_loss = 0.0
        
        # Get model parameters
        A = lds.A
        B = lds.B
        C = lds.C
        M = lds.M
        
        # Compute loss by summing (C.T @ A^i @ B.T + M[:,:,i] - stu_impulse[i])**2 directly
        for i in range(seq_len):
            # Compute C @ A^i @ B directly for the impulse response at time i
            # This is equivalent to computing the impulse response at time i
            x = B.T
            x = (A**i).reshape(-1,1) * x
            y_pred = C.T @ x
            
            # Add M[:,:,i] for the first kx steps
            if i < kx:
                y_pred = y_pred + M[:,:,i]
            
            # Compute squared error with stu_impulse[i]
            squared_error = torch.sum((y_pred - stu_impulse[i])**2)
            running_loss += squared_error
        
        # Compute mean squared error
        total_loss = running_loss / seq_len
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(lds.parameters(), max_norm=1)
        optimizer.step()
    
        with torch.no_grad():
            lds.A.data.clamp_(max=1, min=-1)
    
        # Store loss and print every 10 epochs
        if epoch % 10 == 0:
            loss_value = total_loss.item()
            lds_loss_values.append(loss_value)
            print(f"Epoch {epoch}, Loss: {loss_value}")
            
            # Save current model if it's the best so far
            if loss_value < best_loss:
                best_loss = loss_value
                torch.save(lds.state_dict(), f"{args.output_dir}/lds_best.pth")
    
    # Save final model
    torch.save(lds.state_dict(), f"{args.output_dir}/lds_final.pth")
    
    # Save loss values to a file
    np.save(f"{args.output_dir}/loss_values.npy", np.array(lds_loss_values))
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, epochs, 10), lds_loss_values, marker='o')
    plt.title('LDS Training Loss (every 10 epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f"{args.output_dir}/training_loss.png")
    
    # Compute the impulse response of the trained LDS model
    print("Generating LDS impulse response for evaluation...")
    with torch.no_grad():
        lds_impulse = lds.impulse(seq_len=stu_impulse.shape[0])
    
    # Print shapes for verification
    print(f"LDS impulse shape: {lds_impulse.shape}")
    print(f"STU impulse shape: {stu_impulse.shape}")
    
    # Compute the mean squared error between the two impulse responses
    mse = torch.mean((lds_impulse - stu_impulse) ** 2)
    print(f"Mean Squared Error between LDS and STU impulse: {mse.item()}")
    
    # Visualize a few impulse responses for comparison
    # Select a few input-output pairs to visualize
    input_idx = 10  # First input dimension
    output_indices = [0, 1]  # First two output dimensions
    
    plt.figure(figsize=(12, 8))
    for i, output_idx in enumerate(output_indices):
        plt.subplot(len(output_indices), 1, i+1)
        
        # Plot LDS impulse response
        plt.plot(lds_impulse[:, output_idx, input_idx].cpu().numpy(), 
                 label=f'LDS Impulse (out={output_idx}, in={input_idx})')
        
        # Plot STU impulse response
        plt.plot(stu_impulse[:, output_idx, input_idx].cpu().numpy(), 
                 label=f'STU Impulse (out={output_idx}, in={input_idx})')
        
        plt.title(f'Impulse Response: Output {output_idx}, Input {input_idx}')
        plt.xlabel('Time step')
        plt.ylabel('Response')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/impulse_comparison.png")
    
    # Test how the models respond to Gaussian input
    print("Testing models with Gaussian input...")
    
    # Generate Gaussian input sequence
    test_seq_len = 1000
    input_dim = stu_impulse.shape[2]  # Get input dimension from the impulse shape
    batch_size = 1
    
    # Create random Gaussian input
    np.random.seed(42)  # For reproducibility
    gaussian_input = torch.tensor(np.random.normal(0, 1, (batch_size, test_seq_len, input_dim)), 
                                 dtype=torch.float32).to(device)
    
    # Run both models on the same input
    with torch.no_grad():
        # Get LDS response to Gaussian input
        lds_response = lds(gaussian_input)
        
        # For STU, we need to use the impulse response to compute the output
        # This is essentially a convolution of the input with the impulse response
        stu_response = torch.zeros((batch_size, test_seq_len, stu_impulse.shape[1]), 
                                  dtype=torch.float32).to(device)
        
        # Convolve input with impulse response
        for b in range(batch_size):
            for t in range(test_seq_len):
                for tau in range(min(t+1, stu_impulse.shape[0])):
                    stu_response[b, t] += torch.matmul(
                        stu_impulse[tau], gaussian_input[b, t-tau]
                    )
    
    # Compute MSE between responses
    response_mse = torch.mean((lds_response - stu_response) ** 2)
    print(f"MSE between LDS and STU responses to Gaussian input: {response_mse.item()}")
    
    # Visualize a few output dimensions
    output_indices = [0, 1]  # First two output dimensions
    
    plt.figure(figsize=(12, 8))
    for i, output_idx in enumerate(output_indices):
        plt.subplot(len(output_indices), 1, i+1)
        
        # Plot LDS response
        plt.plot(lds_response[0, :, output_idx].cpu().numpy(), 
                 label=f'LDS Response (out={output_idx})')
        
        # Plot STU response
        plt.plot(stu_response[0, :, output_idx].cpu().numpy(), 
                 label=f'STU Response (out={output_idx})')
        
        plt.title(f'Response to Gaussian Input: Output {output_idx}')
        plt.xlabel('Time step')
        plt.ylabel('Response')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/gaussian_response_comparison.png")
    
    print(f"All results saved to {args.output_dir}")
    
    # Save a summary of all parameters and results
    with open(f"{args.output_dir}/summary.txt", "w") as f:
        f.write(f"Layer index: {layer_i}\n")
        f.write(f"State dimension: {state_dim}\n")
        f.write(f"Sequence length: {seq_len}\n")
        f.write(f"KX parameter: {kx}\n")
        f.write(f"Learning rate: {lr}\n")
        f.write(f"Training epochs: {epochs}\n")
        f.write(f"Final training loss: {lds_loss_values[-1]}\n")
        f.write(f"Best training loss: {best_loss}\n")
        f.write(f"Impulse MSE: {mse.item()}\n")
        f.write(f"Gaussian response MSE: {response_mse.item()}\n")

if __name__ == "__main__":
    main()