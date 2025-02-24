import torch
from torch import nn

from lds_utils import exponential_decay_init, compute_ar_x_preds

class LDS(nn.Module):
    def __init__(self, state_dim, input_dim, output_dim, kx=10, lam=5, dtype=torch.float):
        super(LDS, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.dtype = dtype
        
        self.h0 = nn.Parameter(torch.randn(state_dim, dtype=dtype))
    
        self.A = nn.Parameter(exponential_decay_init([state_dim], lam=lam).to(dtype))
        self.B = nn.Parameter((torch.randn(input_dim, state_dim) / input_dim).to(dtype))
        self.C = nn.Parameter((torch.randn(state_dim, output_dim) / state_dim).to(dtype))
        self.M = nn.Parameter((torch.randn(output_dim, input_dim, kx) / input_dim).to(dtype))

    def forward(self, inputs):
        device = inputs.device
        inputs = inputs.to(self.dtype)
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
        return mse_loss(outputs, targets.to(self.dtype))
    

    def impulse(self, seq_len = 1024):
      # Initialize output tensor
      outputs = torch.zeros(seq_len)
      
      # For each position
      for i in range(seq_len):
          # Compute A^i
          a_power = self.A ** i
          
          # Multiply C[:,0] * A^i * B[i]
          outputs[i] = torch.sum(self.C[:,0] * a_power * self.B[0])
          
      return outputs
