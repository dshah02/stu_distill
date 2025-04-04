import torch
from torch import nn

#NO Autoregressive LDS
class NLDS(nn.Module):
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
        self.A = nn.Parameter(torch.randn(state_dim).to(dtype))
        self.B = nn.Parameter((torch.randn(input_dim, state_dim) / input_dim).to(dtype))
        self.C = nn.Parameter((torch.randn(state_dim, output_dim) / state_dim).to(dtype))

        self.h0 = nn.Parameter(torch.zeros(state_dim, dtype=dtype)).cuda()

        # We'll maintain the hidden state 'h' for recurrent generation.
        self.h = self.h0.unsqueeze(0).expand(bsz_dim, -1).clone().cuda()

    def reset_state(self, batch_size=896):
        """
        Resets the hidden state and AR buffer.
        The hidden state 'h' is set to h0, replicated along the batch dimension.
        """
        self.cache = False
        device = self.A.device
        self.h = self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)
        

    @torch.compile
    def next_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Optimized single-step update:
          1) h_{t+1} = A * h_t + x_t @ B.
          2) lds_out = h_{t+1} @ C.
          3) Update the AR buffer and compute AR output in one optimized step if kx > 0.
        Returns final_out: shape [bsz, output_dim].
        """
        self.h = self.h * self.A + x_t.matmul(self.B)
        lds_out = self.h.matmul(self.C)
        return lds_out
        
    @torch.compile
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
        if seq_len == 1:
            y_t = self.next_step(inputs.squeeze(1))
            return y_t.unsqueeze(1)  # shape => [batch_size, 1, output_dim]
        outputs = []
        for t in range(seq_len):
            x_t = inputs[:, t, :]
            y_t = self.next_step(x_t)
            outputs.append(y_t.unsqueeze(1))
        return torch.cat(outputs, dim=1)


