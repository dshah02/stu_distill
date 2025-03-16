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



# # Let's verify what compute_ar_x_preds is equivalent to
# # We'll manually compute the autoregressive prediction and compare with the function

# # Create a simple test case
# L = 10
# x = torch.randn(1, L, 1)  # Same x as in cell 13
# M = torch.randn(lds_model.M.data.shape)  # Shape: [output_dim, input_dim, kx]

# # Manual computation of autoregressive prediction
# manual_ar_pred = torch.zeros(1, L, M.shape[0], device=x.device)
# for t in range(L):
#     for k in range(min(t+1, M.shape[2])):  # Only use available past inputs
#         if t-k >= 0:  # Make sure we don't go out of bounds
#             manual_ar_pred[0, t] += M[:, :, k] @ x[0, t-k]

# # Using the utility function
# ar_pred_from_function = compute_ar_x_preds(M, x)

# # Compare the results
# print("Manual computation result:")
# print(manual_ar_pred[0, :, 0])
# print("\nFunction computation result:")
# print(ar_pred_from_function[0, :, 0])
# print("\nDifference:")
# print(torch.abs(manual_ar_pred - ar_pred_from_function).max().item())

# # Verify that this is equivalent to convolution with flipped kernel
# # For the first output dimension
# kernel = M[0, 0].flip(dims=[0])  # Flip the kernel for convolution
# conv_result = torch.nn.functional.conv1d(
#     x.transpose(1, 2),  # [1, 1, L]
#     kernel.unsqueeze(0).unsqueeze(0),  # [1, 1, kx]
#     padding=0
# )
# print("\nConvolution result:")
# print(conv_result.squeeze())
