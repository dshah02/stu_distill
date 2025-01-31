import torch
from torch import nn

def exponential_decay_init(size, lam=5.0):
    """
    Samples from an exponential distribution with rate lam, 
    then clips at 1, does (1 - clipped_value),
    and finally multiplies by ±1 with probability 1/2.
    """
    # 1) Sample uniform [0,1], convert to exponential
    u = torch.rand(size)
    x = -1.0 / lam * torch.log(1 - u)  # Exponential(λ = lam)

    # 2) Clip at 1
    x = torch.clamp(x, max=1.0)

    # 3) Subtract from 1 (to be near 1 for small x)
    x = 1.0 - x  # Now we have distribution mostly near 1 for large lam

    # 4) Multiply by ±1 with prob 1/2
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

class LDS(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, kx=10):
        super(LDS, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.h0 = nn.Parameter(torch.randn(state_dim))
        # init_A = torch.randn(state_dim)
        # self.A = nn.Parameter(init_A / torch.max(torch.abs(init_A)))

        # self.A = nn.Parameter((torch.rand(state_dim) * 0.2 + 0.8) * torch.sign(torch.randn(state_dim)))
        self.A = nn.Parameter(exponential_decay_init([state_dim], lam = 15))
        self.B = nn.Parameter(torch.randn(input_dim, state_dim) / input_dim)
        self.C = nn.Parameter(torch.randn(state_dim, output_dim) / state_dim)
        self.M = nn.Parameter(torch.randn(output_dim, input_dim, kx) / (output_dim))

    def forward(self, inputs):
        device = inputs.device
        bsz, seq_len, _ = inputs.shape
        h_t = self.h0.expand(bsz, self.state_dim).to(device)
        A = self.A.flatten()
        all_h_t = []
        for t in range(seq_len):
            u_t = inputs[:, t, :]
            h_t = A * h_t + (u_t @ self.B)
            all_h_t.append(h_t.unsqueeze(1))
        all_h_t = torch.cat(all_h_t, dim=1)
        lds_out = torch.matmul(all_h_t, self.C)

        ar = compute_ar_x_preds(self.M, inputs)
        return lds_out + ar

    def compute_loss(self, inputs, targets):
        mse_loss = nn.MSELoss()
        outputs = self(inputs)
        return mse_loss(outputs, targets)