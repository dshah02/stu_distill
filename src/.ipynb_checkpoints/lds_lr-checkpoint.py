import torch
from torch import nn

from lds import exponential_decay_init, compute_ar_x_preds #maybe move to utils

import math
class LDS_LR(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, kx=5, rank = 50, lam = 1):
        super(LDS_LR, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.h0 = nn.Parameter(torch.randn(state_dim))
 
        self.A = nn.Parameter(exponential_decay_init([state_dim], lam = lam))

        self.B1 = nn.Parameter(torch.randn(input_dim, rank) / input_dim) #each of these ranks could be different
        self.B2   = nn.Parameter(torch.randn(rank, state_dim) / math.sqrt(rank))

        self.C1 = nn.Parameter(torch.randn(state_dim, rank) / state_dim)
        self.C2   = nn.Parameter(torch.randn(rank, output_dim) / math.sqrt(rank))

        self.M2 = nn.Parameter(torch.randn(output_dim, rank, kx) / output_dim)
        self.M1   = nn.Parameter(torch.randn(rank, input_dim, kx) / math.sqrt(rank))
    
    def forward(self, inputs):
        device = inputs.device
        bsz, seq_len, _ = inputs.shape
        h_t = self.h0.expand(bsz, self.state_dim).to(device)
        A = self.A.flatten()
        
        all_h_t = []
        for t in range(seq_len):
            u_t = inputs[:, t, :]
            h_t = A * h_t + ((u_t @ self.B1) @ self.B2)
            all_h_t.append(h_t.unsqueeze(1))
        all_h_t = torch.cat(all_h_t, dim=1)
        lds_out = torch.matmul(all_h_t, self.C1)
        lds_out = torch.matmul(lds_out, self.C2)

        ar = compute_ar_x_preds(self.M1, inputs)
        ar = compute_ar_x_preds(self.M2, ar)
        return lds_out + ar

    def compute_loss(self, inputs, targets):
        mse_loss = nn.MSELoss()
        outputs = self(inputs)
        return mse_loss(outputs, targets)
