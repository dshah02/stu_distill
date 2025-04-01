import torch
from torch import nn

from lds_utils import exponential_decay_init, compute_ar_x_preds


class LDS(nn.Module):
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        kx: int = 5,
        dtype: torch.dtype = torch.float32,
        bsz_dim = 896,
    ):
        """
        state_dim: dimension of LDS hidden state h_t.
        input_dim: dimension of input x_t.
        output_dim: dimension of output.
        kx: AR order (number of taps).
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.dtype = dtype
        self.bsz_dim = bsz_dim
        self.cache = False

        # Note: lam should be defined in your environment or passed as an argument.
        self.A = nn.Parameter(exponential_decay_init([state_dim], lam=5).to(dtype))
        self.B = nn.Parameter((torch.randn(input_dim, state_dim) / input_dim).to(dtype))
        self.C = nn.Parameter((torch.randn(state_dim, output_dim) / state_dim).to(dtype))
        self.M = nn.Parameter((torch.randn(output_dim, input_dim, kx) / input_dim).to(dtype))

        self.h0 = nn.Parameter(torch.zeros(state_dim, dtype=dtype)).cuda()

        # We'll maintain the hidden state 'h' for recurrent generation.
        self.h = self.h0.unsqueeze(0).expand(bsz_dim, -1).clone().cuda()

        if self.kx != 0:
            self.register_buffer("ar_buffer", torch.zeros(bsz_dim, kx, 1, dtype=dtype))

    def reset_state(self, batch_size=896):
        """
        Resets the hidden state and AR buffer.
        The hidden state 'h' is set to h0, replicated along the batch dimension.
        """
        self.cache = False
        device = self.A.device
        self.h = self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)
        if self.kx != 0:
            self.ar_buffer = torch.zeros(batch_size, self.kx, 1, dtype=self.dtype, device=device)

    @torch.compile
    def next_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Optimized single-step update:
          1) h_{t+1} = A * h_t + x_t @ B.
          2) lds_out = h_{t+1} @ C.
          3) Update the AR buffer and compute AR output in one optimized step if kx > 0.
        Returns final_out: shape [bsz, output_dim].
        """
        # Compute LDS update and output in one step
        self.h = self.h * self.A + x_t.matmul(self.B)
        lds_out = self.h.matmul(self.C)
        
        # # Early return if no AR component
        if self.kx == 0:
            return lds_out
            
        # Update AR buffer
        self.ar_buffer = torch.roll(self.ar_buffer, shifts=1, dims=1)
        self.ar_buffer[:, 0, :] = x_t
        
        # Calculate AR output using einsum for optimal performance
        # This avoids both explicit loops and problematic reshaping
        # einsum notation: [batch, k, d_in], [d_out, d_in, k] -> [batch, d_out]
        ar_out = torch.einsum('bki,oik->bo', self.ar_buffer, self.M)
        
        return lds_out + ar_out
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the LDS model.
    
        Args:
            inputs (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
    
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, output_dim].
        """
        
        batch_size, seq_len, _ = inputs.size()
        # Reset the hidden state and AR buffer for a new sequence.
        if not self.cache:
            self.reset_state(batch_size)
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            y_t = self.next_step(x_t)
            outputs.append(y_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)