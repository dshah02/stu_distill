import torch
from torch import nn

def exponential_decay_init(size, lam=5.0):
    """
    Samples from an exponential distribution with rate lam, 
    ensuring the log does not exceed 1 by restricting u's range.
    Then does (1 - value), and finally multiplies by ±1 with probability 1/2.
    """
    # 1) Sample uniform [0, 1 - exp(-lam)], convert to exponential
    max_u = 1 - torch.exp(torch.tensor(-lam))
    u = torch.rand(size) * max_u
    x = -1.0 / lam * torch.log(1 - u)  # Exponential(λ = lam)

    # 2) Subtract from 1 (to be near 1 for small x)
    x = 1.0 - x  # Now we have distribution mostly near 1 for large lam

    # 3) Multiply by ±1 with prob 1/2
    sign = torch.sign(torch.randn(size))
    return x * sign
    
def compute_ar_x_preds(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    d_out, d_in, k = w.shape
    b, l, d_in_x = x.shape
    assert d_in == d_in_x, (
        f"Dimension mismatch: w.shape={w.shape}, x.shape={x.shape}"
    )

    o = torch.einsum("oik,bli->bklo", w, x)

    for i in range(k):
        o[:, i] = torch.roll(o[:, i], shifts=i, dims=1)

    m = torch.triu(torch.ones(k, l, dtype=o.dtype, device=o.device))  # [k, l]
    m = m.unsqueeze(-1).repeat(1, 1, d_out)  # [k, l, d_out]

    ar_x_preds = torch.sum(o * m, dim=1)  # now shape is [b, l, d_out]

    return ar_x_preds