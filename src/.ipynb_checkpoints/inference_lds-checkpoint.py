import torch
from torch import nn

from lds_utils import exponential_decay_init, compute_ar_x_preds

# @torch.compile
def single_step_ar(M: torch.Tensor, ar_buffer: torch.Tensor) -> torch.Tensor:
    """
    Single-step AR with ring buffer of size k => O(k) each step
    """
    bsz, k, d_in = ar_buffer.shape
    d_out, d_in2, k2 = M.shape
    assert (k == k2) and (d_in == d_in2), "Dimension mismatch"
    ar_out = torch.zeros((bsz, d_out), device=ar_buffer.device, dtype=ar_buffer.dtype)
    for i in range(k):
        ar_out += ar_buffer[:, i, :].matmul(M[:, :, i].T)
    return ar_out


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
        state_dim: dimension of LDS hidden state h_t
        input_dim: dimension of input x_t
        output_dim: dimension of output
        kx: AR order (number of taps)
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.dtype = dtype
        self.bsz_dim = bsz_dim

        # Parameters:
        # A is diagonal => store as shape [state_dim]
        self.A = nn.Parameter(torch.zeros(state_dim, dtype=dtype))
        self.B = nn.Parameter(torch.zeros(input_dim, state_dim, dtype=dtype))
        self.C = nn.Parameter(torch.zeros(state_dim, output_dim, dtype=dtype))
        # AR kernel: shape [d_out, d_in, k]
        self.M = nn.Parameter(torch.zeros(output_dim, input_dim, kx, dtype=dtype))
        self.h0 = nn.Parameter(torch.randn(state_dim, dtype=dtype))

        # We'll maintain the hidden state 'h' for recurrent generation
        self.h = self.h0.unsqueeze(0).expand(bsz_dim, -1).clone()

        # We'll also maintain a ring buffer for the AR term
        self.register_buffer("ar_buffer", torch.zeros(bsz_dim, kx, 1, dtype=dtype))

        # No gradients in inference scenario
        for param in self.parameters():
            param.requires_grad_(False)

    def reset_state(self, batch_size = 896):
        """
        Resets the hidden state and AR buffer. The hidden state 'h' is set to h0,
        replicated along the batch dimension.
        """
        device = self.A.device
        # h0 is [state_dim]. Unsqueeze to shape [1, state_dim],
        # then expand to [batch_size, state_dim] and clone to get an independent copy.
        self.h = self.h0.unsqueeze(0).expand(batch_size, -1).clone()
        
        # Reset the AR buffer accordingly (adjust dimensions as needed).
        self.ar_buffer = torch.zeros(batch_size, self.kx, 1, dtype=self.dtype, device=device)

    # @torch.compile
    def next_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Single-step update:
          1) h_{t+1} = A*h_t + x_t @ B
          2) lds_out = h_{t+1} @ C
          3) ar_out = single_step_ar(M, ar_buffer)
             shift ar_buffer to include x_t as the newest input
          4) final_out = lds_out + ar_out
        Returns final_out: shape [bsz, output_dim].
        """
        with torch.no_grad(): #DO THIS LATER
            # 1) LDS recurrent update
            # self.h shape: [bsz, state_dim]
            # self.A shape: [state_dim]
            # x_t shape: [bsz, input_dim]
            # B shape: [input_dim, state_dim]
            self.h = self.h * self.A + x_t.matmul(self.B)  # (bsz, state_dim)

            # 2) Multiply by C => (bsz, output_dim)
            lds_out = self.h.matmul(self.C)

            # 3) AR from last k inputs
            ar_out = single_step_ar(self.M, self.ar_buffer)

            # 4) Shift the AR buffer
            # We want the newest input to go into position 0,
            # and old positions to move down by 1.
            # simplest approach is to roll:
            self.ar_buffer = torch.roll(self.ar_buffer, shifts=1, dims=1)
            # now place x_t in index 0
            self.ar_buffer[:, 0, :] = x_t

            return lds_out + ar_out

    def generate(self, x_init: torch.Tensor, steps: int = 10) -> torch.Tensor:
        """
        Example multi-step generation:
          - x_init: shape [bsz, input_dim]
          - steps: number of tokens/frames to generate
        Returns a tensor of shape [bsz, steps, output_dim].
        
        Typically you'd feed each new output back in as 'x_t' 
        (like a language model token feedback).
        But here we just demonstrate the recurrent LDS+AR step repeated 'steps' times.
        """
        outputs = []
        x_t = x_init
        for _ in range(steps):
            y_t = self.next_step(x_t)  # shape [bsz, output_dim]
            outputs.append(y_t.unsqueeze(1))

            # Example feedback: if you want the next input to be the last output 
            # (like next token = model's guess), you'd do x_t = y_t or some transform.
            # For demonstration, let's just set x_t = y_t here.
            x_t = y_t  # shape [bsz, output_dim], might mismatch input_dim => your logic needed

        return torch.cat(outputs, dim=1)  # [bsz, steps, output_dim]

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
        self.reset_state(batch_size)
    
        outputs = []
        # Iterate over each time step in the sequence.
        for t in range(seq_len):
            # Extract the input at time t: shape [batch_size, input_dim].
            x_t = inputs[:, t, :]
            # Compute the output for this time step.
            y_t = self.next_step(x_t)
            # Append the output with a new sequence dimension.
            outputs.append(y_t.unsqueeze(1))
        
        # Concatenate outputs along the sequence dimension.
        return torch.cat(outputs, dim=1)