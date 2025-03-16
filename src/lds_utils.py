import torch
from torch import nn

def exponential_decay_init(size, lam=5.0):
    """
    Samples from an exponential distribution with rate lam, 
    ensuring the log does not exceed 1 by restricting u's range.
    Then does (1 - value), and finally multiplies by ±1 with probability 1/2.
    """
    # 1) Sample uniform [0, 1 - exp(-lam)], convert to exponential
    max_u = 1 - torch.exp(-torch.tensor(lam))
    u = torch.rand(size) * max_u
    x = -1.0 / lam * torch.log(1 - u)  # Exponential(λ = lam)

    # 2) Subtract from 1 (to be near 1 for small x)
    x = 1.0 - x  # Now we have distribution mostly near 1 for large lam

    # 3) Multiply by ±1 with prob 1/2
    sign = torch.sign(torch.randn(size))
    return x * sign
    
def compute_ar_x_preds(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute autoregressive predictions using weights w and inputs x.
    
    Args:
        w: Tensor of shape (d_out, d_in, kx) containing AR weights
        x: Tensor of shape (batch_size, seq_len, d_in) containing input sequences
        
    Returns:
        Tensor of shape (batch_size, seq_len, d_out) containing AR predictions
    """
    batch_size, seq_len, d_in = x.shape
    d_out, d_in_w, kx = w.shape
    assert d_in == d_in_w, f"Dimension mismatch: w.shape={w.shape}, x.shape={x.shape}"
    
    # Initialize output tensor
    ar_pred = torch.zeros(batch_size, seq_len, d_out, device=x.device)
    
    # For each time step
    for t in range(seq_len):
        # For each lag in the AR model
        for k in range(min(t+1, kx)):  # Only use available past inputs
            if t-k >= 0:  # Make sure we don't go out of bounds
                # Get the input at time t-k for all batches
                x_t_minus_k = x[:, t-k, :]  # Shape: [batch_size, d_in]
                
                # Get the weights for lag k
                w_k = w[:, :, k]  # Shape: [d_out, d_in]
                
                # Compute the matrix multiplication using torch operations
                ar_pred[:, t, :] += x_t_minus_k @ w_k.t()
    
    return ar_pred

def compute_ar_x_preds_hard(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Compute autoregressive predictions using weights w and inputs x.
    
    Args:
        w: Tensor of shape (d_out, d_in, kx) containing AR weights
        x: Tensor of shape (batch_size, seq_len, d_in) containing input sequences
        
    Returns:
        Tensor of shape (batch_size, seq_len, d_out) containing AR predictions
    """
    batch_size, seq_len, d_in = x.shape
    d_out, d_in_w, kx = w.shape
    assert d_in == d_in_w, f"Dimension mismatch: w.shape={w.shape}, x.shape={x.shape}"
    
    # Initialize output tensor
    ar_pred = torch.zeros(batch_size, seq_len, d_out, device=x.device).to(w.dtype)
    
    # For each time step
    for t in range(seq_len):
        # For each lag in the AR model
        for k in range(min(t+1, kx)):  # Only use available past inputs
            if t-k >= 0:  # Make sure we don't go out of bounds
                ar_pred[:, t] += torch.einsum('oi,bi->bo', w[:, :, k], x[:, t-k])
    
    return ar_pred


def compute_ar_x_preds_lr(w_down: torch.Tensor, w_up: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    b, l, d_in = x.shape
    rank, d_in_w, k = w_down.shape
    d_out, rank_w, k_w = w_up.shape
    assert d_in == d_in_w, f"Dimension mismatch: x.shape={x.shape}, w_down.shape={w_down.shape}"
    assert rank == rank_w and k == k_w, f"Dimension mismatch between w_down and w_up: {w_down.shape} vs {w_up.shape}"
    
    o = torch.einsum("ork,rik,bli->bklo", w_up, w_down, x)
    
    for i in range(k):
        o[:, i] = torch.roll(o[:, i], shifts=i, dims=1)
    
    m = torch.triu(torch.ones(k, l, dtype=o.dtype, device=o.device))
    m = m.unsqueeze(-1).expand(k, l, d_out)
    ar_x_preds = torch.sum(o * m, dim=1)
    return ar_x_preds