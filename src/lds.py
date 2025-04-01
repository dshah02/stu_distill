import torch
from torch import nn

from lds_utils import exponential_decay_init, compute_ar_x_preds

class LDS(nn.Module):
    def __init__(
        self, 
        state_dim, 
        input_dim, 
        output_dim, 
        kx=10, 
        lam=5, 
        dtype=torch.float, 
        cache=False
    ):
        super(LDS, self).__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.dtype = dtype
        self.cache = cache
        
        self.h0 = nn.Parameter(torch.zeros(state_dim, dtype=dtype))
    
        self.A = nn.Parameter(exponential_decay_init([state_dim], lam=lam).to(dtype))
        self.B = nn.Parameter((torch.randn(input_dim, state_dim) / input_dim).to(dtype))
        self.C = nn.Parameter((torch.randn(state_dim, output_dim) / state_dim).to(dtype))
        self.M = nn.Parameter((torch.randn(output_dim, input_dim, kx) / input_dim).to(dtype))
        
        # Initialize hidden state if caching is used
        self.h = None

    def forward(self, inputs):
        device = inputs.device
        inputs = inputs.to(self.dtype)
        bsz, seq_len, _ = inputs.shape
        
        # Use cached hidden state if enabled and available.
        if self.cache:
            # If self.h is not set or the batch size has changed, initialize it.
            if self.h is None or self.h.shape[0] != bsz:
                self.h = self.h0.expand(bsz, self.state_dim).to(device)
            h_t = self.h
        else:
            h_t = self.h0.expand(bsz, self.state_dim).to(device)
            
        A = self.A.flatten()
        all_h_t = []
        
        for t in range(seq_len):
            u_t = inputs[:, t, :]
            h_t = A * h_t + (u_t @ self.B)
            all_h_t.append(h_t.unsqueeze(1))
        
        # Update the maintained hidden state if caching is enabled.
        if self.cache:
            self.h = h_t.detach()
            
        all_h_t = torch.cat(all_h_t, dim=1)
        lds_out = torch.matmul(all_h_t, self.C)

        if self.kx > 0:
            ar = compute_ar_x_preds(self.M, inputs)
            return lds_out + ar
        return lds_out

    def compute_loss(self, inputs, targets):
        mse_loss = nn.MSELoss()
        outputs = self(inputs)
        return mse_loss(outputs, targets.to(self.dtype))
    
    def generate_trajectory(self, us, h0=None):
        """
        Generate a trajectory of observations given a sequence of inputs.
        
        Args:
            us: Tensor of shape (seq_len, input_dim) containing input sequence
            h0: Optional initial hidden state. If None, use the model's h0
            
        Returns:
            Tensor of shape (seq_len, output_dim) containing output sequence
        """
        _, d_u = us.shape
        assert d_u == self.input_dim, (d_u, self.input_dim)
        
        if h0 is not None:
            h_t = h0
        else:
            h_t = self.h0
            
        A = self.A.flatten()
        obs = []
        
        for u in us:
            h_t = A * h_t + (u @ self.B)
            o_t = h_t @ self.C
            obs.append(o_t)
        
        obs = torch.stack(obs, dim=0)
        
        if self.kx > 0:
            obs += compute_ar_x_preds(self.M, us.unsqueeze(0))
            
        return obs

    def impulse(self, seq_len=1024):
        """
        Compute the impulse response of the LDS for all input-output pairs.
        
        Args:
            seq_len: Length of the impulse response
            
        Returns:
            outputs: Tensor of shape (seq_len, d_out, d_in) containing impulse responses
        """
        # Get dimensions from the LDS matrices
        d_h = self.A.shape[0]  # state dimension
        d_in = self.B.shape[0]  # input dimension
        d_out = self.C.shape[1]  # output dimension
        
        # Initialize output tensor
        outputs = torch.zeros(seq_len, d_out, d_in, device=self.A.device)
        
        # For each time step
        for t in range(seq_len):
            # Compute A^t
            a_power = self.A ** t  # Shape: (d_h,)
            
            # For each input dimension
            for i in range(d_in):
                # Get the i-th column of B (input to state mapping)
                b_i = self.B[i, :]  # Shape: (d_h,)
                
                # Compute a_power * b_i (element-wise multiplication)
                state_response = a_power * b_i  # Shape: (d_h,)
                
                # Compute C @ state_response for all outputs
                output_response = self.C.T @ state_response  # Shape: (d_out,)
                
                # Store in the output tensor
                outputs[t, :, i] = output_response
        # Add the contribution from the direct input-to-output mapping (M matrix)
        if self.kx > 0:
            # For each time step
            for t in range(self.kx):
                outputs[t,:,:] += self.M[:,:, t]
               
        return outputs