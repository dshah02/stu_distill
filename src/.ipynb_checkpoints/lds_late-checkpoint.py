import torch
from torch import nn

from lds_utils import exponential_decay_init, compute_ar_x_preds

#this doesn't work well
class LDS_Late(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, kx=10, lam = 5):
        super(LDS_Late, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.h0 = nn.Parameter(torch.randn(state_dim))
    
        self.A = nn.Parameter(exponential_decay_init([state_dim], lam = lam))
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

        new_lds_out = torch.zeros_like(lds_out)
        new_lds_out[:, :-self.kx, :] = lds_out[:, self.kx:, :] #shifting by kx
        ar = compute_ar_x_preds(self.M, inputs)

        return ar + new_lds_out

    def compute_loss(self, inputs, targets):
        mse_loss = nn.MSELoss()
        outputs = self(inputs)
        return mse_loss(outputs, targets)