import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Assume compute_ar_x_preds is imported if needed.
# from lds_utils import compute_ar_x_preds

class RecurrentLDS_Step(nn.Module):
    """
    A single-step recurrent LDS module that computes the next output given a single time-step input.
    
    It maintains internal hidden state and an autoregressive (AR) buffer.
    """
    def __init__(self, lds, M_inputs, M_filters):
        """
        Args:
          lds: LDS instance with attributes A, B, C, h0 already set.
          M_phi_plus, M_phi_minus: spectral weights of shape [K, d_in, d_out].
        """
        super(RecurrentLDS_Step, self).__init__()
        self.lds = lds
        self.dtype = self.lds.A.dtype
        self.M_inputs= self.M_inputs.data.to(self.dtype)
        self.M_filters = self.M_inputs.data.to(self.dtype)
        
        
        # Internal states (to be initialized via reset_state)
        self.hidden_state = None   # shape: [B, hidden_dim]
        self.ar_buffer = None      # shape: [B, k_u, d_in]
    
    def reset_state(self, batch_size):
        """
        Resets the hidden state and AR buffer for a new sequence.
        Here, we use the size of self.lds.A (number of poles) for hidden state.
        """
        n_poles = self.lds.A.shape[0]  # Force use A's length (e.g. 216)
        self.hidden_state = torch.zeros(batch_size, n_poles, dtype=self.lds.A.dtype, 
                                          device=self.lds.A.device)
        if self.M.shape[-1] > 0:
            k_u = self.M.shape[-1]
            # Assuming input dimension is 1.
            self.ar_buffer = torch.zeros(batch_size, k_u, 1, dtype=self.lds.A.dtype, 
                                           device=self.lds.A.device)
    
    def step(self, x_t):
        """
        Processes a single time step.
        
        Args:
          x_t: Input tensor of shape [B, 1] (assuming input_dim=1).
        
        Returns:
          y_t: Output tensor of shape [B, d_out].
          
        Updates the internal hidden state and AR buffer.
        """
        B, _ = x_t.shape
        # If hidden state exists but has wrong size, reset it.
        if (self.hidden_state is None) or (self.hidden_state.shape[1] != self.lds.A.shape[0]):
            self.reset_state(B)
        
        # 1) Update hidden state: h = h * A + B * x_t.
        self.hidden_state = self.hidden_state * self.lds.A + x_t * self.lds.B  # [B, hidden_dim]
        
        # 2) Compute raw LDS output: y_raw = h @ C, shape [B, 48] (assuming output_dim=48).
        y_raw = torch.matmul(self.hidden_state, self.lds.C)  # [B, 48]
        
        # 3) Split into plus and minus halves (each of dimension K, e.g. 24)
        half_dim = y_raw.shape[1] // 2
        out_plus = y_raw[:, :half_dim]   # [B, K]
        out_minus = y_raw[:, half_dim:]  # [B, K]
        
        # 4) Expand dims for contraction: [B, K, 1]
        out_plus_3d = out_plus.unsqueeze(-1)
        out_minus_3d = out_minus.unsqueeze(-1)
        
        # 5) Contract with spectral weights (match dtype to x_t)
        plus_expand = torch.einsum('bki,kid->bid', out_plus_3d, self.M_phi_plus.to(x_t.dtype))
        minus_expand = torch.einsum('bki,kid->bid', out_minus_3d, self.M_phi_minus.to(x_t.dtype))
        y_spec = plus_expand.sum(dim=1) + minus_expand.sum(dim=1)  # [B, d_out]
        
        # 6) AR component: update AR buffer and compute AR prediction if enabled.
        if self.M.shape[-1] > 0:
            self.ar_buffer = torch.roll(self.ar_buffer, shifts=-1, dims=1)
            self.ar_buffer[:, -1, :] = x_t  # store new input at end.
            ar_pred = compute_ar_x_preds(self.M, self.ar_buffer)  # [B, d_out]
            y_t = y_spec + ar_pred
        else:
            y_t = y_spec
        
        return y_t